# triagegeist

Predicting emergency department triage acuity using machine learning — classifying patients into ESI urgency levels from structured clinical data collected at the point of triage.

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
| **Ensemble (LGBM + XGB tuned)** | **0.9730** | Best; `submissions/ensemble_lgbm_xgb_a0.25_0.9730.csv` |
| LightGBM (default) | 0.9727 | 5-fold OOF; `submissions/lightgbm_0.9730.csv` |
| XGBoost (tuned) | 0.9723 | 5-fold OOF after Optuna; `submissions/best_params_xgb.json` |
| XGBoost (default) | 0.9705 ± 0.0014 | Untuned baseline |
| MLP (PyTorch) | ~0.9694 | Single holdout (20% split) |

All tree models use **5-fold stratified CV** (OOF macro F1). The ensemble blends LightGBM and tuned XGBoost probability outputs at α=0.25/0.75.

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
├── train_all.py                   ← baseline: trains all models, saves figures + best submission CSV
├── tune.py                        ← Optuna tuning entry point (LGBM / XGB)
├── ensemble.py                    ← OOF blend of default LGBM + tuned XGB
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
│   ├── tuning.py                  ← Optuna objectives + study runner
│   ├── utils.py                   ← metrics, plotting, SEED, color palette
│   └── tests/
│       └── test_feature_engineering.py
├── submissions/
│   ├── best_params_lgbm.json      ← best Optuna params for LightGBM
│   ├── best_params_xgb.json       ← best Optuna params for XGBoost
│   ├── ensemble_summary.json      ← blend strategy + final OOF score
│   └── *.csv                      ← submission files
└── figures/                       ← auto-saved plots from train_all.py
```

---

## Feature Engineering

**297 features total** — built in `src/feature_engineering.py` via `build_features(train_df, test_df, history_df, chief_df)`.

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


## Models

All models are defined in `src/models.py` and importable into the notebook.

### LightGBM (`run_lgbm`)
- 5-fold stratified CV, retrain on full data for test predictions
- Default: 300 estimators, max_depth=7, lr=0.1, subsample=0.8, colsample_bytree=0.8
- **CV Macro F1: 0.9727**

### XGBoost — tuned (`tune.py --model xgb`)
- Tuned via Optuna (50 trials, 3-fold CV, CUDA); best params in `submissions/best_params_xgb.json`
- Best params: max_depth=6, lr=0.050, n_estimators=1992, gamma=1.87, subsample=0.73
- **CV Macro F1: 0.9723** (5-fold OOF; +0.0018 over untuned baseline of 0.9705)

### Ensemble (`ensemble.py`)
- OOF probability blend: 25% LightGBM + 75% tuned XGBoost
- Blend weight optimised by grid search (α ∈ [0, 1], step 0.05) on 5-fold OOF
- **OOF Macro F1: 0.9730**; submission: `submissions/ensemble_lgbm_xgb_a0.25_0.9730.csv`

### MLP — PyTorch (`run_nn`)
- Architecture: Linear → BatchNorm → ReLU → Dropout(0.3), layers [256, 128, 64]
- Single 80/20 stratified holdout; Adam optimizer, CrossEntropyLoss
- **Holdout Macro F1: ~0.9694**

---

## Hyperparameter Tuning

XGBoost was tuned with [Optuna](https://optuna.org/) using TPE sampling and HyperbandPruner. LightGBM tuning (20 trials) did not improve over the default configuration.

| | LGBM | XGBoost |
|---|---|---|
| Trials | 20 | 50 |
| CV folds | 5 | 5 |
| Best 5-fold F1 | 0.9727 | 0.9723 |
| n_estimators | 300 | 1992 |

Key XGBoost params found: `max_depth=6`, `learning_rate=0.050`, `gamma=1.87`, `max_delta_step=1`, `colsample_bynode=0.83`.

Run tuning:
```bash
python tune.py --model xgb --n-trials 50   # resumes from SQLite if interrupted
python tune.py --model lgbm --n-trials 20
```

---

## Setup

```bash
# Run all baseline models
python train_all.py

# Hyperparameter tuning (resumes from SQLite if killed mid-run)
python tune.py --model lgbm --n-trials 20
python tune.py --model xgb  --n-trials 50

# Ensemble tuned XGB + default LGBM
python ensemble.py

# Open notebook
conda run -p C:\Users\Xh321\Miniforge3\envs\dsc80 jupyter notebook
```

**Reproducibility:** `SEED = 93` — passed to all models, CV splits, and Optuna samplers.


