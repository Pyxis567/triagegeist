import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import SEED


def build_features(train_df, test_df, history_df, chief_df):
    """
    Full feature engineering pipeline. Fits on train, applies to test.

    Parameters
    ----------
    train_df   : raw train.csv DataFrame (includes triage_acuity)
    test_df    : raw test.csv DataFrame (no triage_acuity)
    history_df : patient_history.csv DataFrame
    chief_df   : chief_complaints.csv DataFrame

    Returns
    -------
    X_train : pd.DataFrame  (features only, no target)
    y_train : pd.Series
    X_test  : pd.DataFrame
    fit_params : dict       (medians, tfidf vectorizer — apply to new data)
    """
    train = train_df.copy()
    test  = test_df.copy()

    target = "triage_acuity"
    y_train = train[target].copy()

    # ── 1. Merge patient_history (25 binary comorbidity flags) ───────────────
    hx_cols = [c for c in history_df.columns if c != "patient_id"]
    train = train.merge(history_df[["patient_id"] + hx_cols], on="patient_id", how="left")
    test  = test.merge(history_df[["patient_id"] + hx_cols],  on="patient_id", how="left")

    # ── 1b. Missingness indicators (MNAR — encode informative absence) ─────────
    # Must be created BEFORE imputation while NaNs are still present.
    # Missingness is not random: lower-acuity patients often skip certain vitals,
    # so "not measured" is itself a signal of triage level.
    # systolic and diastolic always co-occur (same 5.18%); one indicator covers both.
    mnar_cols = {
        "is_missing_bp":               "systolic_bp",
        "is_missing_respiratory_rate": "respiratory_rate",
        "is_missing_temperature_c":    "temperature_c",
    }
    for indicator, source_col in mnar_cols.items():
        train[indicator] = train[source_col].isna().astype(int)
        test[indicator]  = test[source_col].isna().astype(int)

    # ── 2. Impute missing vitals (fit medians on train only) ─────────────────
    impute_cols = [
        "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
        "pulse_pressure", "shock_index", "respiratory_rate", "temperature_c",
    ]
    medians = train[impute_cols].median()
    train[impute_cols] = train[impute_cols].fillna(medians)
    test[impute_cols]  = test[impute_cols].fillna(medians)

    # ════════════════════════════════════════════════════════════════════════
    # EXISTING TRANSFORMS — ported from triagegeist_analysis.ipynb
    # ════════════════════════════════════════════════════════════════════════

    # ── 3. Time-of-day flags (from arrival_hour) ─────────────────────────────
    for df in (train, test):
        hour = df["arrival_hour"]
        df["is_daytime"] = hour.between(8, 17).astype(int)
        df["is_evening"] = hour.between(18, 22).astype(int)
        df["is_night"]   = (~hour.between(8, 22)).astype(int)

    # ── 4. Age binning → one-hot ─────────────────────────────────────────────
    age_bins   = [0, 1, 12, 17, 44, 54, 64, 79, float("inf")]
    age_labels = ["infant", "child", "adolescent", "young_adult",
                  "middle_aged_early", "middle_aged_late", "senior", "elderly"]
    for df in (train, test):
        df["age_group_binned"] = pd.cut(df["age"], bins=age_bins,
                                        labels=age_labels, right=True)
        for label in age_labels:
            df[f"is_{label}"] = (df["age_group_binned"] == label).astype(int)

    # ── 5. One-hot encoding (11 categorical cols) ────────────────────────────
    ohe_cols = [
        "arrival_mode", "arrival_season", "arrival_month", "shift",
        "sex", "language", "insurance_type", "transport_origin",
        "pain_location", "mental_status_triage", "chief_complaint_system",
    ]
    # Fit on train, align test to same columns.
    # Use per-column prefix (is_{col}) to avoid name collisions between source columns
    # that share the same value (e.g. shift="evening" vs manual is_evening flag).
    # Cast to str first so numeric cols like arrival_month (int64) are also encoded
    col_prefixes = [f"is_{col}" for col in ohe_cols]
    train_dummies = pd.get_dummies(train[ohe_cols].astype(str), prefix=col_prefixes,
                                   prefix_sep="_", drop_first=False).astype(int)
    test_dummies  = pd.get_dummies(test[ohe_cols].astype(str),  prefix=col_prefixes,
                                   prefix_sep="_", drop_first=False).astype(int)
    test_dummies  = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    train = pd.concat([train, train_dummies], axis=1)
    test  = pd.concat([test,  test_dummies],  axis=1)

    # ════════════════════════════════════════════════════════════════════════
    # NEW FEATURES — additional engineering
    # ════════════════════════════════════════════════════════════════════════

    # ── 6. Vital signs interaction features ──────────────────────────────────
    # pulse_pressure_ratio: relative pulse pressure (wider = more unstable)
    for df in (train, test):
        df["pulse_pressure_ratio"] = df["pulse_pressure"] / (df["systolic_bp"] + 1e-6)
        # map_hr_product: proxy for cardiac output stress
        df["map_hr_product"] = df["mean_arterial_pressure"] * df["heart_rate"]

    # ── 7. Comorbidity burden score (sum of 25 hx_ flags) ────────────────────
    # Distinct from num_comorbidities in raw data which may reflect different coding
    for df in (train, test):
        df["comorbidity_count"] = df[hx_cols].sum(axis=1)

    # ── 8. Prior utilization ratio ───────────────────────────────────────────
    # Fraction of past ED visits that resulted in admission (severity proxy)
    for df in (train, test):
        df["ed_admission_ratio"] = (
            df["num_prior_admissions_12m"] / (df["num_prior_ed_visits_12m"] + 1)
        )

    # ── 9. NEWS2 clinical risk flags ─────────────────────────────────────────
    # NEWS2 ≥ 7 → high risk (national early warning score threshold)
    # NEWS2 5–6 → medium risk
    for df in (train, test):
        df["is_high_news2"]   = (df["news2_score"] >= 7).astype(int)
        df["is_medium_news2"] = df["news2_score"].between(5, 6).astype(int)

    # ── 10. Age × NEWS2 interaction ──────────────────────────────────────────
    # Elderly patients with high NEWS2 disproportionately escalate to L1/L2
    for df in (train, test):
        df["age_news2"] = df["age"] * df["news2_score"]

    # ════════════════════════════════════════════════════════════════════════
    # CLINICALLY-GROUNDED FEATURES (informed by NEWS2, ESI, qSOFA, SIRS)
    # Reference: NEWS2 Royal College of Physicians 2017; ESI v5 AHRQ;
    #            qSOFA Singer et al. JAMA 2016; SIRS ACCP/SCCM 1992
    # ════════════════════════════════════════════════════════════════════════

    # ── 11. NEWS2 individual component sub-scores ────────────────────────────
    # Each vital maps to 0-3 points per the NEWS2 scoring table.
    # LightGBM's top features are all raw vitals — giving it the validated
    # break-points explicitly should sharpen its splits.
    for df in (train, test):
        rr  = df["respiratory_rate"]
        sp  = df["spo2"]
        sbp = df["systolic_bp"]
        hr  = df["heart_rate"]
        tmp = df["temperature_c"]

        # np.select evaluates conditions in order; first True wins.
        df["news2_rr_score"] = np.select(
            [rr <= 8,  rr <= 11, rr <= 20, rr <= 24],
            [3,        1,        0,         2],
            default=3,
        )
        df["news2_spo2_score"] = np.select(
            [sp <= 91, sp <= 93, sp <= 95],
            [3,        2,        1],
            default=0,
        )
        df["news2_sbp_score"] = np.select(
            [sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219],
            [3,         2,          1,           0],
            default=3,
        )
        df["news2_hr_score"] = np.select(
            [hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130],
            [3,        1,        0,        1,          2],
            default=3,
        )
        df["news2_temp_score"] = np.select(
            [tmp <= 35.0, tmp <= 36.0, tmp <= 38.0, tmp <= 39.0],
            [3,           1,           0,            1],
            default=2,
        )
        df["news2_max_component"] = df[[
            "news2_rr_score", "news2_spo2_score", "news2_sbp_score",
            "news2_hr_score", "news2_temp_score",
        ]].max(axis=1)
        # Any single parameter scoring 3 triggers escalation in NEWS2 protocol
        df["is_any_news2_max3"] = (df["news2_max_component"] == 3).astype(int)

    # ── 12. qSOFA score (sepsis early warning) ───────────────────────────────
    # 3-item bedside score: RR ≥22, altered mentation (GCS<15), sBP ≤100.
    # Score ≥2 → 3–14× increased in-hospital mortality (Singer et al. 2016).
    for df in (train, test):
        df["qsofa_rr"]  = (df["respiratory_rate"] >= 22).astype(int)
        df["qsofa_sbp"] = (df["systolic_bp"] <= 100).astype(int)
        df["qsofa_gcs"] = (df["gcs_total"] < 15).astype(int)
        df["qsofa_score"]      = df["qsofa_rr"] + df["qsofa_sbp"] + df["qsofa_gcs"]
        df["is_qsofa_positive"] = (df["qsofa_score"] >= 2).astype(int)

    # ── 13. ESI Level-2 vital-sign threshold flags ───────────────────────────
    # These are the exact vital-sign danger thresholds used in ESI v5 to
    # distinguish Level 2 (emergent) from Level 3 (urgent).
    for df in (train, test):
        df["is_tachycardia"]        = (df["heart_rate"] > 100).astype(int)
        df["is_bradycardia"]        = (df["heart_rate"] < 50).astype(int)
        df["is_hypotension"]        = (df["systolic_bp"] < 90).astype(int)
        df["is_hypertensive_crisis"] = (df["systolic_bp"] >= 220).astype(int)
        df["is_hypoxia"]            = (df["spo2"] < 92).astype(int)
        df["is_severe_hypoxia"]     = (df["spo2"] <= 91).astype(int)
        df["is_tachypnea"]          = (df["respiratory_rate"] > 20).astype(int)
        df["is_bradypnea"]          = (df["respiratory_rate"] <= 8).astype(int)
        df["is_fever"]              = (df["temperature_c"] > 38.3).astype(int)
        df["is_hypothermia"]        = (df["temperature_c"] <= 35.0).astype(int)
        df["is_hyperthermia"]       = (df["temperature_c"] >= 39.1).astype(int)
        df["is_gcs_impaired"]       = (df["gcs_total"] < 15).astype(int)
        df["is_high_pain"]          = (df["pain_score"] >= 7).astype(int)
        df["is_severe_pain"]        = (df["pain_score"] >= 9).astype(int)

    # ── 14. Partial SIRS criteria (WBC unavailable) ──────────────────────────
    # SIRS ≥2/4 criteria = systemic inflammatory response.
    # Without WBC we can score on temp, HR, RR (3 of 4 criteria).
    for df in (train, test):
        sirs_temp = ((df["temperature_c"] < 36) | (df["temperature_c"] > 38)).astype(int)
        sirs_hr   = (df["heart_rate"] > 90).astype(int)
        sirs_rr   = (df["respiratory_rate"] > 20).astype(int)
        df["sirs_count"]       = sirs_temp + sirs_hr + sirs_rr
        df["is_sirs_positive"] = (df["sirs_count"] >= 2).astype(int)

    # ── 15. Composite clinical crisis patterns ───────────────────────────────
    for df in (train, test):
        # SpO2 <94 + RR >20 together signal overt respiratory compromise
        df["is_respiratory_compromise"] = (
            (df["spo2"] < 94) & (df["respiratory_rate"] > 20)
        ).astype(int)
        # Septic shock triad: temperature dysregulation + tachycardia + hypotension
        df["is_septic_shock_pattern"] = (
            ((df["temperature_c"] > 38) | (df["temperature_c"] < 36)) &
            (df["heart_rate"] > 90) &
            (df["systolic_bp"] < 90)
        ).astype(int)
        # Shock index ≥1.0 = uncompensated shock (HR > sBP)
        df["is_shock_severe"] = (df["shock_index"] >= 1.0).astype(int)

    # ── 16. Cross-score interaction features ─────────────────────────────────
    for df in (train, test):
        df["pain_news2"]        = df["pain_score"]        * df["news2_score"]
        df["gcs_news2"]         = df["gcs_total"]         * df["news2_score"]
        df["comorbidity_news2"] = df["comorbidity_count"] * df["news2_score"]
        df["age_qsofa"]         = df["age"]               * df["qsofa_score"]
        # SpO2/RR ratio — drops when both hypoxia and tachypnea coexist
        df["spo2_rr_ratio"]     = df["spo2"] / (df["respiratory_rate"] + 1)
        # High pain coinciding with sepsis-risk flag
        df["high_pain_qsofa"]   = df["is_high_pain"] * df["is_qsofa_positive"]

    # ── 17. Age-stratified features ──────────────────────────────────────────
    # Pediatric and geriatric patients follow different physiology:
    # - Children have higher baseline HR/RR; the adult thresholds in step 13
    #   would incorrectly flag a normal infant HR of 130 as "tachycardia".
    # - Elderly patients have attenuated stress responses; even mild NEWS2
    #   elevation is disproportionately dangerous.
    for df in (train, test):
        age = df["age"]

        # Consolidated group flags (useful as clean interaction bases)
        df["is_pediatric"]   = (age < 18).astype(int)
        df["is_geriatric"]   = (age > 64).astype(int)   # senior + elderly

        # ── Age-adjusted tachycardia/bradycardia (PALS/AHA norms) ────────────
        # Normal HR upper limits by age group
        hr_upper = np.select(
            [age <= 1, age <= 5, age <= 12, age <= 17],
            [160,      150,      120,        100],
            default=100,
        )
        # Normal HR lower limits by age group
        hr_lower = np.select(
            [age <= 1, age <= 5, age <= 12, age <= 17],
            [100,       90,       70,         60],
            default=60,
        )
        df["is_age_adj_tachycardia"] = (df["heart_rate"] > hr_upper).astype(int)
        df["is_age_adj_bradycardia"] = (df["heart_rate"] < hr_lower).astype(int)

        # ── Age-adjusted tachypnea (WHO/AAP respiratory norms) ───────────────
        rr_upper = np.select(
            [age <= 1, age <= 3, age <= 12, age <= 17],
            [60,       40,       30,         20],
            default=20,
        )
        df["is_age_adj_tachypnea"] = (df["respiratory_rate"] > rr_upper).astype(int)

        # ── REMS age component (Olsson et al. 2004, validated in ED) ─────────
        # Age contributes 0/2/3/5/6 points in the Rapid Emergency Medicine Score
        df["rems_age_score"] = np.select(
            [age < 45, age < 55, age < 65, age < 75],
            [0,        2,        3,         5],
            default=6,
        )

        # ── Age × comorbidity burden ──────────────────────────────────────────
        df["age_comorbidity"]          = age * df["comorbidity_count"]
        df["geriatric_comorbidity"]    = df["is_geriatric"] * df["comorbidity_count"]
        # Elderly with ≥3 comorbidities = high frailty proxy
        df["elderly_high_comorbidity"] = df["is_elderly"] * (df["comorbidity_count"] >= 3).astype(int)

        # ── Geriatric-specific danger flags ───────────────────────────────────
        # Altered mental status in elderly → high escalation risk
        df["elderly_gcs_impaired"]  = df["is_elderly"]   * df["is_gcs_impaired"]
        # Geriatric patients with high NEWS2 → disproportionate mortality risk
        df["geriatric_news2"]       = df["is_geriatric"] * df["news2_score"]
        df["geriatric_high_news2"]  = df["is_geriatric"] * df["is_high_news2"]
        # Elderly + qSOFA positive → very high sepsis mortality
        df["elderly_qsofa"]         = df["is_elderly"]   * df["qsofa_score"]

        # ── Pediatric-specific danger flags ───────────────────────────────────
        # Fever in infants <1 year is always high acuity
        df["infant_fever"]         = df["is_infant"]    * df["is_fever"]
        df["pediatric_fever"]      = df["is_pediatric"] * df["is_fever"]
        df["pediatric_high_news2"] = df["is_pediatric"] * df["is_high_news2"]

        # ── Age × pain (elderly underreport pain; high pain score = more alarming) ──
        df["age_pain"]          = age * df["pain_score"]
        df["elderly_high_pain"] = df["is_elderly"] * df["is_high_pain"]

    # ════════════════════════════════════════════════════════════════════════
    # TF-IDF ON chief_complaint_raw (Task 6)
    # ════════════════════════════════════════════════════════════════════════

    # ── 11. Merge chief complaint text then vectorize ─────────────────────────
    train = train.merge(chief_df[["patient_id", "chief_complaint_raw"]],
                        on="patient_id", how="left")
    test  = test.merge(chief_df[["patient_id", "chief_complaint_raw"]],
                       on="patient_id", how="left")
    train["chief_complaint_raw"] = train["chief_complaint_raw"].fillna("")
    test["chief_complaint_raw"]  = test["chief_complaint_raw"].fillna("")

    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2),
                            stop_words="english")
    tfidf.fit(train["chief_complaint_raw"])

    # Replace spaces with underscores so XGBoost/LGBM accept the column names
    tfidf_cols = [f"tfidf_{t}".replace(" ", "_") for t in tfidf.get_feature_names_out()]
    tfidf_train = pd.DataFrame(
        tfidf.transform(train["chief_complaint_raw"]).toarray(),
        columns=tfidf_cols, index=train.index,
    )
    tfidf_test = pd.DataFrame(
        tfidf.transform(test["chief_complaint_raw"]).toarray(),
        columns=tfidf_cols, index=test.index,
    )
    train = pd.concat([train, tfidf_train], axis=1)
    test  = pd.concat([test,  tfidf_test],  axis=1)

    # ── Drop columns ─────────────────────────────────────────────────────────
    leakage_train = ["disposition", "ed_los_hours"]
    always_drop   = [
        "patient_id", "triage_nurse_id", "arrival_day", "site_id",
        "arrival_hour", "age", "age_group", "age_group_binned", "language",
        "chief_complaint_raw",
    ] + ohe_cols

    train.drop(columns=leakage_train + always_drop + [target],
               errors="ignore", inplace=True)
    test.drop(columns=always_drop, errors="ignore", inplace=True)

    # Keep only numeric columns
    X_train = train.select_dtypes(include="number")
    X_test  = test.select_dtypes(include="number")

    # Align test to train column set
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    fit_params = {"medians": medians, "tfidf": tfidf, "ohe_columns": train_dummies.columns.tolist()}

    return X_train, y_train, X_test, fit_params
