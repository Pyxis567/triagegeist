# triagegeist — CLAUDE.md

## Project Overview

Kaggle competition: predict the **Emergency Severity Index (ESI) triage acuity level** (1–5) for ED patients from structured clinical data collected at triage.

- **Kaggle page:** https://www.kaggle.com/competitions/triagegeist
- **Task:** 5-class classification
- **Evaluation metric:** Macro F1-score (all five classes weighted equally)
- **Submission format:** `patient_id, triage_acuity` — see `sample_submission/sample_submission.csv`

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `raw_data/train.csv` | 80,000 | Labeled records (40 features + `triage_acuity` target) |
| `raw_data/test.csv` | — | Unlabeled records for Kaggle submission |
| `raw_data/chief_complaints.csv` | — | Free-text chief complaint per patient |
| `raw_data/patient_history.csv` | — | 25 binary comorbidity flags per patient |

Feature groups: **vital signs** (BP, HR, SpO2, temp, RR), **demographics** (age, sex, insurance), **clinical scores** (NEWS2, GCS, pain), **arrival context** (mode, time, shift), **prior utilization** (ED visits/admissions past 12 months).

## Repository Layout

```
triagegeist/
├── CLAUDE.md                      ← this file
├── README.md
├── triagegeist.ipynb              ← main integration + visualization notebook
├── triagegeist_analysis.ipynb     ← legacy scratch notebook (gitignored, do not use)
├── raw_data/
│   ├── train.csv
│   ├── test.csv
│   ├── chief_complaints.csv
│   └── patient_history.csv
├── sample_submission/
│   └── sample_submission.csv
├── src/                           ← Python modules (import into notebook)
│   ├── feature_engineering.py     ← all feature transforms
│   ├── models.py                  ← model definitions / training helpers
│   └── utils.py                   ← shared helpers (metrics, plotting)
└── submissions/                   ← generated CSVs ready to upload
```

Create `src/` and `submissions/` when you first need them; they are not committed until populated.

## Code Organization Rules

- **Logic lives in `src/`** — feature engineering, model classes, utilities go in Python modules so they can be tested and reused.
- **The notebook is the single source of truth for output** — every result, plot, and submission must be produced by importing from `src/` and running cells in `triagegeist.ipynb`. Do not duplicate logic inside notebook cells.
- **Keep notebook cells thin** — one import + one call per cell section. Heavy computation belongs in `src/`.
- When adding a new model, add it to `src/models.py`, then add a section in the notebook under `## Models` following the existing `run_xgb` pattern.

## Agent-Skills Workflow

Use the agent-skills suite for all non-trivial work. The full cycle is:

```
/spec  →  /plan  →  /build  →  /test  →  /review  →  /code-simplify  →  /ship
```

| Skill | When to invoke |
|-------|---------------|
| `/spec` | Before starting any new model or feature engineering block — write the spec first |
| `/plan` | Break the spec into ordered, verifiable tasks before writing code |
| `/build` | Implement one task at a time; build, verify, commit incrementally |
| `/test` | After each build step — unit-test `src/` modules (not notebook cells) |
| `/review` | Before merging any change or submitting to Kaggle — five-axis review |
| `/code-simplify` | After the feature works — reduce complexity without changing behavior |
| `/ship` | Final pre-submission checklist: metric verified, notebook runs clean top-to-bottom, CSV saved |

Never skip `/review` before generating a Kaggle submission CSV.

## Color & Style Requirements

All visualizations must use this palette consistently. Do not substitute other colors.

### Triage level colors (use in this order: Level 1 → Level 5)
```python
ACUITY_COLORS = ["#D7191C", "#F46D43", "#FDAE61", "#A6D96A", "#1A9641"]
# Level 1 Resuscitation  →  #D7191C  (dark red)
# Level 2 Emergent       →  #F46D43  (orange-red)
# Level 3 Urgent         →  #FDAE61  (orange-yellow)
# Level 4 Less Urgent    →  #A6D96A  (light green)
# Level 5 Non-Urgent     →  #1A9641  (dark green)
```

Use `ACUITY_COLORS` for:
- Per-class bar charts (F1 by triage level, class distribution)
- Any chart where bars/segments map directly to ESI levels
- Confusion matrix row/column annotations if colored

### Confusion matrix — exact style
Reproduce this call exactly for every confusion matrix. Do not change any of these parameters.
```python
sns.heatmap(
    cm_pct, annot=True, fmt=".1f", cmap="Blues",
    xticklabels=labels, yticklabels=labels,
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "%"},
)
ax.set_title("Confusion Matrix (%)", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
```
where `cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100` and `labels = [f"L{i}" for i in sorted(classes)]`.

### Feature importance bars — exact style
```python
# ranked low → high, top-N slice, then colormap
fi_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n_features))
importances.plot(kind="barh", ax=ax, color=fi_colors)
ax.axvline(importances.mean(), color="red", linestyle="--", linewidth=1, label="mean")
```

### Plot defaults
```python
# Apply at the top of every notebook section that produces plots
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "#e5e5e5",
})
# For Plotly figures
PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
)
```

### Acuity label strings
```python
ACUITY_LABELS = {
    1: "L1\nResuscitation",
    2: "L2\nEmergent",
    3: "L3\nUrgent",
    4: "L4\nLess Urgent",
    5: "L5\nNon-Urgent",
}
```

Move these constants into `src/utils.py` and import them everywhere — never redefine them inline.

## Evaluation & Submission

```python
from sklearn.metrics import f1_score
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
```

- Target: **maximize macro F1** — minority classes (L1, L5) matter as much as L3.
- Always report per-class F1 alongside macro F1 in every model section.
- Save submission CSVs to `submissions/<model_name>_<macro_f1>.csv`.
- Run `/ship` checklist before every Kaggle upload.

## Random Seeds & Reproducibility

```python
SEED = 93
np.random.seed(SEED)
# Pass random_state=SEED to all sklearn / XGBoost models
# For PyTorch: torch.manual_seed(SEED)
```

## GPU Usage

Always prefer GPU over CPU for model training. Detect and assign device at the top of every training script/section:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- **PyTorch models:** move model and tensors to `device` — `model.to(device)`, `X_tensor.to(device)`.
- **XGBoost:** set `tree_method="hist", device="cuda"` when a GPU is available:
  ```python
  xgb_params = {
      "tree_method": "hist",
      "device": "cuda" if torch.cuda.is_available() else "cpu",
  }
  ```
- Never hardcode `device="cpu"` or `"cuda"` — always use the detection snippet above.

## Current State (as of 2026-06-06)

### Models
Random Forest has been removed. Active models are XGBoost, LightGBM, and MLP (PyTorch). XGBoost and LightGBM both use **5-fold stratified CV**.

| Model | CV Macro F1 | Notes |
|-------|-------------|-------|
| **Ensemble** | **0.9730** | Best; 25% LGBM + 75% tuned XGB OOF blend; `submissions/ensemble_lgbm_xgb_a0.25_0.9730.csv` |
| LightGBM (default) | 0.9727 | 5-fold OOF; `submissions/lightgbm_0.9730.csv` |
| XGBoost (tuned) | 0.9723 | Optuna 50 trials; `submissions/best_params_xgb.json` |
| XGBoost (default) | 0.9705 | Untuned baseline |
| MLP | ~0.9694 | Single holdout |

### Hyperparameter Tuning (completed 2026-06-06)
- **Tool:** Optuna, TPE sampler, HyperbandPruner, SQLite persistence (`submissions/optuna_*.db`)
- **CV:** 3-fold during search, validated with 5-fold after
- **LGBM:** 20 trials — best 3-fold F1=0.9695; default params proved stronger in 5-fold (0.9727 vs 0.9699 tuned). Default params kept.
- **XGBoost:** 50 trials — best 3-fold F1=0.9716, validated 5-fold F1=0.9723 (+0.0018 over untuned). Best params: `max_depth=6, lr=0.050, n_estimators=1992, gamma=1.87, max_delta_step=1, colsample_bynode=0.83`.

### Ensemble (completed 2026-06-06)
Grid-searched blend weight α ∈ [0, 1] (step 0.05) on 5-fold OOF probabilities:
- Best: **α=0.25** (25% default LGBM + 75% tuned XGB) → OOF F1 = **0.9730**
- Submission: `submissions/ensemble_lgbm_xgb_a0.25_0.9730.csv`

### Feature Engineering (297 features total)
Built in `src/feature_engineering.py`. Key additions beyond raw features:

- **Missingness indicators** (`is_missing_bp`, `is_missing_respiratory_rate`, `is_missing_temperature_c`) — created before imputation to preserve the MNAR signal
- **NEWS2 sub-scores** — individual component scores (0–3) for RR, SpO2, sBP, HR, temp; plus `news2_max_component` and `is_any_news2_max3`
- **qSOFA** — `qsofa_score` and `is_qsofa_positive` (sepsis screening; score ≥2 → high mortality risk)
- **ESI Level-2 threshold flags** — tachycardia, bradycardia, hypotension, hypoxia, fever, GCS impairment, high/severe pain, etc.
- **Partial SIRS** — `sirs_count` and `is_sirs_positive` (temp + HR + RR criteria; WBC unavailable)
- **Composite crisis patterns** — `is_respiratory_compromise`, `is_septic_shock_pattern`, `is_shock_severe`
- **Cross-score interactions** — `pain_news2`, `gcs_news2`, `comorbidity_news2`, `spo2_rr_ratio`, `high_pain_qsofa`
- **Age-stratified features** — `is_pediatric`, `is_geriatric`, age-adjusted tachycardia/bradycardia/tachypnea (PALS norms), REMS age score, geriatric risk interactions, pediatric fever flags

### Feature Selection (tested, not applied)
Ran `feature_selection_test.py` — four strategies tested against the 297-feature baseline (LightGBM 5-fold CV). No strategy improved meaningfully; differences were within fold noise (±0.0014). Decision: keep all 297 features. Three zero-importance features identified (`is_hypertensive_crisis`, `is_bradypnea`, `is_hypothermia`) — rare events with no useful splits in this dataset.

## To-Do

- [x] **Hyperparameter tuning** — Optuna on LightGBM and XGBoost; completed 2026-06-06
  - **Setup:** Create `src/tuning.py` with one Optuna objective per model. Both use full **5-fold stratified CV** per trial (same folds as baseline, SEED=93). Use early stopping (`callbacks=[lgb.early_stopping(50)]` / `early_stopping_rounds=50`) so `n_estimators` is found automatically — set it high (3000) and let early stopping terminate. Save the best params + best n_estimators for each model to `submissions/best_params_lgbm.json` and `submissions/best_params_xgb.json` after tuning.

  - **LightGBM tuning — ~3.5 hours, ~200 trials (GPU)**
    - Sampler: TPE; Pruner: HyperbandPruner (prunes unpromising trials early via per-fold intermediate values)
    - Search space:
      - `num_leaves`: 31–1000 (key lever — directly controls model capacity)
      - `max_depth`: -1 (unlimited) or 4–15; try both via categorical
      - `learning_rate`: 0.005–0.2 (log scale)
      - `n_estimators`: 3000 (always set high; early stopping finds true best)
      - `min_child_samples`: 5–300 (prevents over-fitting on small leaves)
      - `subsample` (bagging_fraction): 0.4–1.0
      - `subsample_freq` (bagging_freq): 1–7
      - `colsample_bytree` (feature_fraction): 0.4–1.0
      - `reg_alpha` (L1): 1e-8–10.0 (log)
      - `reg_lambda` (L2): 1e-8–10.0 (log)
      - `min_split_gain`: 0.0–1.0
      - `path_smooth`: 0.0–1.0
    - Expected per-trial time: ~40–60s on RTX 3070 → 200 trials ≈ 2.5–3 hours
    - Objective: mean 5-fold OOF macro F1 (maximize)

  - **XGBoost tuning — ~3 hours, ~100 trials (GPU, `device="cuda"`)**
    - Sampler: TPE; Pruner: HyperbandPruner
    - Search space:
      - `max_depth`: 3–12
      - `min_child_weight`: 1–50
      - `learning_rate`: 0.005–0.2 (log)
      - `n_estimators`: 3000 (with early stopping 50 rounds)
      - `subsample`: 0.4–1.0
      - `colsample_bytree`: 0.4–1.0
      - `colsample_bylevel`: 0.4–1.0
      - `colsample_bynode`: 0.4–1.0
      - `gamma`: 0–5
      - `reg_alpha` (L1): 1e-8–10.0 (log)
      - `reg_lambda` (L2): 1e-8–10.0 (log)
      - `max_delta_step`: 0–10 (can help with class imbalance)
    - Expected per-trial time: ~60–90s on RTX 3070 → 100 trials ≈ 1.5–2.5 hours
    - Objective: mean 5-fold OOF macro F1 (maximize)

  - **Validate winners (~15 min):** For each model, reload best params from JSON, run full 5-fold CV, print per-fold and mean F1. Compare to baselines (LGBM 0.9730, XGB 0.9705). Record gains.

- [x] **Ensemble tuned LGBM + XGBoost** — completed 2026-06-06; OOF F1=0.9730
  - Collect OOF `predict_proba` (shape: n_train × 5) from both tuned models
  - **Strategy 1 — Weighted average:** sweep LGBM weight α ∈ [0, 1] (step 0.05) on OOF, pick α maximizing macro F1; apply same weight to held-out test probabilities before argmax
  - **Strategy 2 — Rank averaging:** average softmax probability ranks across both models
  - **Strategy 3 — Stacking:** train a LogisticRegression meta-learner on the 10 OOF probability columns (5 from each model); use 5-fold CV to generate meta-features to avoid leakage
  - Pick best ensemble strategy by OOF macro F1; generate test submission
  - Save submission as `submissions/ensemble_<strategy>_<f1>.csv`
  - Only submit to Kaggle if OOF gain > fold noise (~0.0015)

- [ ] **CV for MLP** — currently on single holdout; add 5-fold CV for a fair comparison
- [ ] **Submit to Kaggle** — current best: `submissions/ensemble_lgbm_xgb_a0.25_0.9730.csv`
