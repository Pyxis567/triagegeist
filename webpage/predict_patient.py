"""
Single-patient feature transform. Mirrors all 18 stages of
src/feature_engineering.py but operates on one patient dict.
"""
import numpy as np
import pandas as pd
from datetime import datetime

_OHE_COLS     = ["arrival_mode", "arrival_season", "arrival_month", "shift",
                  "sex", "language", "insurance_type", "transport_origin",
                  "pain_location", "mental_status_triage", "chief_complaint_system"]
_OHE_PREFIXES = [f"is_{c}" for c in _OHE_COLS]

_AGE_BINS   = [0, 1, 12, 17, 44, 54, 64, 79, float("inf")]
_AGE_LABELS = ["infant", "child", "adolescent", "young_adult",
               "middle_aged_early", "middle_aged_late", "senior", "elderly"]


def _season(month: int) -> str:
    return {12: "winter", 1: "winter",  2: "winter",
             3: "spring", 4: "spring",  5: "spring",
             6: "summer", 7: "summer",  8: "summer",
             9: "autumn", 10: "autumn", 11: "autumn"}[month]

def _shift(hour: int) -> str:
    if 6  <= hour < 14: return "morning"
    if 14 <= hour < 22: return "afternoon"
    return "night"

def _age_group(age: float) -> str:
    for label, lo, hi in zip(_AGE_LABELS, _AGE_BINS, _AGE_BINS[1:]):
        if lo < age <= hi:
            return label
    return _AGE_LABELS[-1]

def _n2_rr(rr):
    if rr <= 8:  return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3

def _n2_spo2(sp):
    if sp <= 91: return 3
    if sp <= 93: return 2
    if sp <= 95: return 1
    return 0

def _n2_sbp(sbp):
    if sbp <= 90:  return 3
    if sbp <= 100: return 2
    if sbp <= 110: return 1
    if sbp <= 219: return 0
    return 3

def _n2_hr(hr):
    if hr <= 40:  return 3
    if hr <= 50:  return 1
    if hr <= 90:  return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3

def _n2_temp(t):
    if t <= 35.0: return 3
    if t <= 36.0: return 1
    if t <= 38.0: return 0
    if t <= 39.0: return 1
    return 2


def transform_patient(patient: dict, fit_params: dict) -> pd.DataFrame:
    """Apply full feature pipeline to one patient dict → 1-row DataFrame."""

    medians     = fit_params["medians"]
    tfidf       = fit_params["tfidf"]
    ohe_columns = fit_params["ohe_columns"]   # list from training
    hx_cols     = fit_params["hx_cols"]
    feat_cols   = fit_params["feature_columns"]

    p = dict(patient)   # local copy

    # ── Auto-fill time fields from current timestamp ─────────────────────────
    now   = datetime.now()
    hour  = now.hour
    month = now.month
    p.setdefault("arrival_hour",          hour)
    p.setdefault("arrival_month",         month)
    p.setdefault("arrival_season",        _season(month))
    p.setdefault("shift",                 _shift(hour))
    p.setdefault("arrival_day",           now.strftime("%A"))
    p.setdefault("language",              "English")
    p.setdefault("chief_complaint_system","other")
    p.setdefault("site_id",               "SITE-TMP-01")
    p.setdefault("patient_id",            "WEB-000001")
    p.setdefault("triage_nurse_id",       "NURSE-0000")
    p.setdefault("age_group",             "adult")
    p.setdefault("num_comorbidities",     0)

    for col in hx_cols:
        p.setdefault(col, 0)
    p.setdefault("chief_complaint_raw", "")

    # ── Parse raw vitals ─────────────────────────────────────────────────────
    def _f(key, default=np.nan):
        v = p.get(key)
        if v is None or v == "": return default
        try:    return float(v)
        except: return default

    age  = _f("age")
    sbp  = _f("systolic_bp")
    dbp  = _f("diastolic_bp")
    hr   = _f("heart_rate")
    rr   = _f("respiratory_rate")
    tmp  = _f("temperature_c")
    sp   = _f("spo2")
    gcs  = _f("gcs_total")
    pain = _f("pain_score")
    wt   = _f("weight_kg")
    ht   = _f("height_cm")

    # ── Compute derived vitals before imputation ─────────────────────────────
    map_ = ((sbp + 2 * dbp) / 3)      if (np.isfinite(sbp) and np.isfinite(dbp)) else np.nan
    pp   = (sbp - dbp)                 if (np.isfinite(sbp) and np.isfinite(dbp)) else np.nan
    si   = (hr / sbp)                  if (np.isfinite(hr)  and np.isfinite(sbp) and sbp > 0) else np.nan
    bmi  = (wt / (ht / 100) ** 2)     if (np.isfinite(wt)  and np.isfinite(ht)  and ht > 0) else np.nan

    # NEWS2 score (computed from vitals; hospitals measure this at triage)
    n2  = 0
    if np.isfinite(rr):   n2 += _n2_rr(rr)
    if np.isfinite(sp):   n2 += _n2_spo2(sp)
    if np.isfinite(sbp):  n2 += _n2_sbp(sbp)
    if np.isfinite(hr):   n2 += _n2_hr(hr)
    if np.isfinite(tmp):  n2 += _n2_temp(tmp)
    if np.isfinite(gcs) and gcs < 15: n2 += 3

    p.update(dict(
        mean_arterial_pressure=map_, pulse_pressure=pp,
        shock_index=si, bmi=bmi, news2_score=n2,
    ))

    df = pd.DataFrame([p])

    # Ensure numeric types for key columns
    for col in ["age", "systolic_bp", "diastolic_bp", "heart_rate",
                "respiratory_rate", "temperature_c", "spo2", "gcs_total",
                "pain_score", "mean_arterial_pressure", "pulse_pressure",
                "shock_index", "bmi", "news2_score",
                "num_prior_ed_visits_12m", "num_prior_admissions_12m",
                "num_active_medications", "num_comorbidities",
                "weight_kg", "height_cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Stage 1b: MNAR indicators (before imputation) ────────────────────────
    df["is_missing_bp"]               = int(not np.isfinite(sbp))
    df["is_missing_respiratory_rate"] = int(not np.isfinite(rr))
    df["is_missing_temperature_c"]    = int(not np.isfinite(tmp))

    # ── Stage 2: Impute with saved train medians ─────────────────────────────
    impute_cols = ["systolic_bp", "diastolic_bp", "mean_arterial_pressure",
                   "pulse_pressure", "shock_index", "respiratory_rate", "temperature_c"]
    for col in impute_cols:
        if col not in df.columns or not np.isfinite(float(df[col].iloc[0])):
            df[col] = float(medians.get(col, 0))

    # Re-read imputed values as floats
    sbp  = float(df["systolic_bp"].iloc[0])
    dbp  = float(df["diastolic_bp"].iloc[0])
    map_ = float(df["mean_arterial_pressure"].iloc[0])
    pp   = float(df["pulse_pressure"].iloc[0])
    si   = float(df["shock_index"].iloc[0])
    rr   = float(df["respiratory_rate"].iloc[0]) if np.isfinite(rr) else float(medians.get("respiratory_rate", 16))
    tmp  = float(df["temperature_c"].iloc[0])    if np.isfinite(tmp) else float(medians.get("temperature_c", 37))
    # re-use already-computed news2; fill hr/sp/gcs/pain with medians if missing
    hr   = hr   if np.isfinite(hr)   else 80.0
    sp   = sp   if np.isfinite(sp)   else 98.0
    gcs  = gcs  if np.isfinite(gcs)  else 15.0
    pain = pain if np.isfinite(pain) else 0.0
    df["heart_rate"]      = hr
    df["spo2"]            = sp
    df["gcs_total"]       = gcs
    df["pain_score"]      = pain
    df["age"]             = age

    # ── Stage 3: Time-of-day flags ───────────────────────────────────────────
    df["is_daytime"] = int(8 <= hour <= 17)
    df["is_evening"] = int(18 <= hour <= 22)
    df["is_night"]   = int(not (8 <= hour <= 22))

    # ── Stage 4: Age binning → OHE flags ────────────────────────────────────
    grp = _age_group(age)
    df["age_group_binned"] = grp
    for label in _AGE_LABELS:
        df[f"is_{label}"] = int(grp == label)

    # ── Stage 5: OHE — initialise all training columns to 0, then set match ─
    for col in ohe_columns:
        df[col] = 0
    ohe_set = set(ohe_columns)
    for raw_col, prefix in zip(_OHE_COLS, _OHE_PREFIXES):
        val = str(p.get(raw_col, "nan"))
        candidate = f"{prefix}_{val}"
        if candidate in ohe_set:
            df[candidate] = 1

    # ── Stage 6: Vital sign interactions ─────────────────────────────────────
    df["pulse_pressure_ratio"] = pp / (sbp + 1e-6)
    df["map_hr_product"]       = map_ * hr

    # ── Stage 7: Comorbidity burden ───────────────────────────────────────────
    cc = sum(int(p.get(c, 0) or 0) for c in hx_cols)
    df["comorbidity_count"] = cc

    # ── Stage 8: Prior utilisation ratio ─────────────────────────────────────
    prior_ed  = float(p.get("num_prior_ed_visits_12m",  0) or 0)
    prior_adm = float(p.get("num_prior_admissions_12m", 0) or 0)
    df["ed_admission_ratio"] = prior_adm / (prior_ed + 1)

    # ── Stage 9: NEWS2 flags ──────────────────────────────────────────────────
    df["is_high_news2"]   = int(n2 >= 7)
    df["is_medium_news2"] = int(5 <= n2 <= 6)

    # ── Stage 10: Age × NEWS2 ─────────────────────────────────────────────────
    df["age_news2"] = age * n2

    # ── Stage 11: NEWS2 sub-scores ────────────────────────────────────────────
    rr_s   = _n2_rr(rr)
    spo2_s = _n2_spo2(sp)
    sbp_s  = _n2_sbp(sbp)
    hr_s   = _n2_hr(hr)
    tmp_s  = _n2_temp(tmp)
    max_c  = max(rr_s, spo2_s, sbp_s, hr_s, tmp_s)

    df["news2_rr_score"]      = rr_s
    df["news2_spo2_score"]    = spo2_s
    df["news2_sbp_score"]     = sbp_s
    df["news2_hr_score"]      = hr_s
    df["news2_temp_score"]    = tmp_s
    df["news2_max_component"] = max_c
    df["is_any_news2_max3"]   = int(max_c == 3)

    # ── Stage 12: qSOFA ───────────────────────────────────────────────────────
    qrr  = int(rr >= 22)
    qsbp = int(sbp <= 100)
    qgcs = int(gcs < 15)
    qs   = qrr + qsbp + qgcs
    df["qsofa_rr"]         = qrr
    df["qsofa_sbp"]        = qsbp
    df["qsofa_gcs"]        = qgcs
    df["qsofa_score"]      = qs
    df["is_qsofa_positive"] = int(qs >= 2)

    # ── Stage 13: ESI L2 threshold flags ──────────────────────────────────────
    df["is_tachycardia"]         = int(hr  > 100)
    df["is_bradycardia"]         = int(hr  < 50)
    df["is_hypotension"]         = int(sbp < 90)
    df["is_hypertensive_crisis"] = int(sbp >= 220)
    df["is_hypoxia"]             = int(sp  < 92)
    df["is_severe_hypoxia"]      = int(sp  <= 91)
    df["is_tachypnea"]           = int(rr  > 20)
    df["is_bradypnea"]           = int(rr  <= 8)
    df["is_fever"]               = int(tmp > 38.3)
    df["is_hypothermia"]         = int(tmp <= 35.0)
    df["is_hyperthermia"]        = int(tmp >= 39.1)
    df["is_gcs_impaired"]        = int(gcs < 15)
    df["is_high_pain"]           = int(pain >= 7)
    df["is_severe_pain"]         = int(pain >= 9)

    # ── Stage 14: Partial SIRS ────────────────────────────────────────────────
    sirs_t = int((tmp < 36) or (tmp > 38))
    sirs_h = int(hr > 90)
    sirs_r = int(rr > 20)
    df["sirs_count"]       = sirs_t + sirs_h + sirs_r
    df["is_sirs_positive"] = int((sirs_t + sirs_h + sirs_r) >= 2)

    # ── Stage 15: Composite crisis patterns ───────────────────────────────────
    df["is_respiratory_compromise"] = int((sp < 94) and (rr > 20))
    df["is_septic_shock_pattern"]   = int(((tmp > 38) or (tmp < 36)) and (hr > 90) and (sbp < 90))
    df["is_shock_severe"]           = int(si >= 1.0)

    # ── Stage 16: Cross-score interactions ────────────────────────────────────
    df["pain_news2"]        = pain * n2
    df["gcs_news2"]         = gcs  * n2
    df["comorbidity_news2"] = cc   * n2
    df["age_qsofa"]         = age  * qs
    df["spo2_rr_ratio"]     = sp   / (rr + 1)
    df["high_pain_qsofa"]   = int(pain >= 7) * int(qs >= 2)

    # ── Stage 17: Age-stratified features ────────────────────────────────────
    is_ped  = int(age < 18)
    is_ger  = int(age > 64)
    is_eld  = int(age > 79)
    is_inf  = int(0 < age <= 1)

    df["is_pediatric"] = is_ped
    df["is_geriatric"] = is_ger

    if   age <= 1:  hr_up, hr_lo = 160, 100
    elif age <= 5:  hr_up, hr_lo = 150,  90
    elif age <= 12: hr_up, hr_lo = 120,  70
    elif age <= 17: hr_up, hr_lo = 100,  60
    else:           hr_up, hr_lo = 100,  60

    df["is_age_adj_tachycardia"] = int(hr > hr_up)
    df["is_age_adj_bradycardia"] = int(hr < hr_lo)

    if   age <= 1:  rr_up = 60
    elif age <= 3:  rr_up = 40
    elif age <= 12: rr_up = 30
    elif age <= 17: rr_up = 20
    else:           rr_up = 20
    df["is_age_adj_tachypnea"] = int(rr > rr_up)

    if   age < 45: rems = 0
    elif age < 55: rems = 2
    elif age < 65: rems = 3
    elif age < 75: rems = 5
    else:          rems = 6
    df["rems_age_score"] = rems

    df["age_comorbidity"]          = age * cc
    df["geriatric_comorbidity"]    = is_ger * cc
    df["elderly_high_comorbidity"] = is_eld * int(cc >= 3)
    df["elderly_gcs_impaired"]     = is_eld * int(gcs < 15)
    df["geriatric_news2"]          = is_ger * n2
    df["geriatric_high_news2"]     = is_ger * int(n2 >= 7)
    df["elderly_qsofa"]            = is_eld * qs
    df["infant_fever"]             = is_inf * int(tmp > 38.3)
    df["pediatric_fever"]          = is_ped * int(tmp > 38.3)
    df["pediatric_high_news2"]     = is_ped * int(n2 >= 7)
    df["age_pain"]                 = age  * pain
    df["elderly_high_pain"]        = is_eld * int(pain >= 7)

    # ── Stage 18: TF-IDF on chief complaint ───────────────────────────────────
    chief_text = str(p.get("chief_complaint_raw", "") or "")
    tfidf_cols = [f"tfidf_{t}".replace(" ", "_") for t in tfidf.get_feature_names_out()]
    tfidf_vals = tfidf.transform([chief_text]).toarray()[0]
    for col, val in zip(tfidf_cols, tfidf_vals):
        df[col] = float(val)

    # ── Align to training feature columns ─────────────────────────────────────
    result = df.reindex(columns=feat_cols, fill_value=0)
    result = result.apply(pd.to_numeric, errors="coerce").fillna(0)
    return result
