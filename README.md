# triagegeist

Predicting emergency department triage acuity using machine learning — classifying patients into ESI urgency levels from structured clinical data collected at the point of triage.

> [!WARNING]
> 🚧 **This project is under active construction.** Expect incomplete sections, breaking changes, and work-in-progress code.

---

## Competition

**[Triagegeist — Kaggle](https://www.kaggle.com/competitions/triagegeist)**

The challenge is to predict the **Emergency Severity Index (ESI) triage acuity level** assigned to a patient upon arrival at the emergency department. Given structured clinical and demographic data collected at triage, the model must classify each patient into one of five urgency levels:

| Level | Label | Description |
|-------|-------|-------------|
| 1 | Resuscitation | Immediately life-threatening |
| 2 | Emergent | High risk, should not wait |
| 3 | Urgent | Stable but requires multiple resources |
| 4 | Less Urgent | Stable, requires one resource |
| 5 | Non-Urgent | Stable, no resources needed |

**Evaluation metric:** Macro F1-score across all five classes.

---

## Results

| Model | CV Macro F1 | Notes |
|-------|-------------|-------|
| **LightGBM** | **0.9730** | Best; `submissions/lightgbm_0.9730.csv` |
| LightGBM | 0.9724 | Earlier run; `submissions/lightgbm_cv0.9724.csv` |
| XGBoost | 0.9705 ± 0.0014 | 5-fold CV OOF |
| MLP (PyTorch) | ~0.9694 | Single holdout (20% split) |

Both XGBoost and LightGBM use **5-fold stratified CV** — OOF macro F1 is reported, then the model is retrained on the full training set for test predictions.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `raw_data/train.csv` | 80,000 | Labeled patient records (40 features + target) |
| `raw_data/test.csv` | 20,000 | Unlabeled patient records for submission |
| `raw_data/chief_complaints.csv` | — | Raw free-text chief complaint per patient |
| `raw_data/patient_history.csv` | — | 25 binary comorbidity flags per patient |

Key feature groups: **vital signs** (BP, HR, SpO2, temp, respiratory rate), **demographics** (age, sex, insurance), **clinical scores** (NEWS2, GCS, pain score), **arrival context** (mode, time, shift), and **prior utilization** (ED visits and admissions in past 12 months).

---

## Repository Layout

```
triagegeist/
├── CLAUDE.md                      ← agent workflow and style guide
├── README.md
├── train_all.py                   ← runs all models and saves best submission
├── triagegeist.ipynb              ← main integration + visualization notebook
├── raw_data/
│   ├── train.csv
│   ├── test.csv
│   ├── chief_complaints.csv
│   └── patient_history.csv
├── sample_submission/
│   └── sample_submission.csv
├── src/
│   ├── feature_engineering.py     ← all feature transforms (build_features)
│   ├── models.py                  ← run_xgb, run_lgbm, run_nn
│   └── utils.py                   ← metrics, plotting, SEED, color palette
├── submissions/                   ← generated CSVs ready to upload
└── figures/                       ← auto-saved plots from train_all.py
```

---

## Feature Engineering

**297 features total** — built in `src/feature_engineering.py` via `build_features(train_df, test_df, history_df, chief_df)`. Fits on train, applies to test (no leakage).

### Pipeline stages

1. **Merge comorbidities** — join 25 binary flags from `patient_history.csv`
2. **MNAR missingness indicators** — `is_missing_bp`, `is_missing_respiratory_rate`, `is_missing_temperature_c` created before imputation; missingness correlates with triage level
3. **Median imputation** — fit on train only; applied to BP, MAP, pulse pressure, shock index, RR, temperature
4. **Time-of-day flags** — `is_daytime` (8–17), `is_evening` (18–22), `is_night`
5. **Age binning** — 8 groups (infant → elderly), one-hot encoded
6. **One-hot encoding** — 11 categorical columns (arrival mode, season, month, shift, sex, language, insurance, transport origin, pain location, mental status, chief complaint system)
7. **Vital sign interactions** — `pulse_pressure_ratio`, `map_hr_product`
8. **Comorbidity burden** — `comorbidity_count` (sum of 25 hx_ flags)
9. **Prior utilization ratio** — `ed_admission_ratio = admissions / (ed_visits + 1)`
10. **NEWS2 risk flags** — `is_high_news2` (≥7), `is_medium_news2` (5–6), `age_news2`
11. **NEWS2 sub-scores** — individual 0–3 point scores for RR, SpO2, sBP, HR, temp; `news2_max_component`; `is_any_news2_max3`
12. **qSOFA** — `qsofa_score` (RR ≥22 + sBP ≤100 + GCS <15); `is_qsofa_positive` (score ≥2)
13. **ESI Level-2 threshold flags** — tachycardia, bradycardia, hypotension, hypertensive crisis, hypoxia, severe hypoxia, tachypnea, bradypnea, fever, hypothermia, hyperthermia, GCS impairment, high pain, severe pain
14. **Partial SIRS** — `sirs_count` and `is_sirs_positive` (temp + HR + RR; WBC unavailable)
15. **Composite crisis patterns** — `is_respiratory_compromise` (SpO2 <94 + RR >20), `is_septic_shock_pattern`, `is_shock_severe` (shock index ≥1.0)
16. **Cross-score interactions** — `pain_news2`, `gcs_news2`, `comorbidity_news2`, `age_qsofa`, `spo2_rr_ratio`, `high_pain_qsofa`
17. **Age-stratified features** — `is_pediatric`, `is_geriatric`; PALS-adjusted tachycardia/bradycardia/tachypnea thresholds; REMS age score; geriatric danger flags (`geriatric_news2`, `elderly_gcs_impaired`, `elderly_qsofa`); pediatric danger flags (`infant_fever`, `pediatric_high_news2`)
18. **TF-IDF on chief complaint text** — 100 features, unigrams + bigrams, fit on train

### Feature selection (tested, not applied)

Four selection strategies were tested against the 297-feature baseline; none improved meaningfully (all differences within fold noise, ±0.0014). All 297 features are kept. Three features have zero importance in LightGBM (`is_hypertensive_crisis`, `is_bradypnea`, `is_hypothermia`) — rare events with no useful splits in this dataset.

---

## Models

All models are defined in `src/models.py` and importable into the notebook.

### LightGBM (`run_lgbm`)
- 5-fold stratified CV, retrain on full data for test predictions
- Default: 300 estimators, max_depth=7, lr=0.1, subsample=0.8, colsample_bytree=0.8
- GPU training when available (`device="gpu"`)
- **Best CV Macro F1: 0.9730**

### XGBoost (`run_xgb`)
- 5-fold stratified CV, retrain on full data for test predictions
- Default: 300 estimators, max_depth=7, lr=0.1, subsample=0.8, colsample_bytree=0.8
- GPU training when available (`tree_method="hist", device="cuda"`)
- **CV Macro F1: 0.9705 ± 0.0014**

### MLP — PyTorch (`run_nn`)
- Architecture: Linear → BatchNorm → ReLU → Dropout(0.3), layers [256, 128, 64]
- Single 80/20 stratified holdout; Adam optimizer, CrossEntropyLoss
- **Holdout Macro F1: ~0.9694**

---

## Setup

```bash
# Activate environment
conda activate C:\Users\Xh321\Miniforge3\envs\dsc80

# Run all models and save best submission
conda run -p C:\Users\Xh321\Miniforge3\envs\dsc80 python train_all.py

# Or open the notebook
conda run -p C:\Users\Xh321\Miniforge3\envs\dsc80 jupyter notebook
```

**Reproducibility:** `SEED = 93` — passed to all models and CV splits.


