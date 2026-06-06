"""Ensemble tuned XGBoost + best LightGBM via OOF probability blending.

Steps:
  1. Load tuned XGB params from submissions/best_params_xgb.json
  2. Run 5-fold CV for tuned XGB and untuned LGBM -> OOF probabilities
  3. Grid-search blend weight alpha (LGBM*alpha + XGB*(1-alpha)) on OOF
  4. Retrain both models on full data, apply best alpha to test set
  5. Save submission to submissions/ensemble_<label>_<f1>.csv
"""
import sys, os, json
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from src.utils import SEED
from src.feature_engineering import build_features

np.random.seed(SEED)
N_SPLITS = 5
SKF = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_raw = pd.read_csv("raw_data/train.csv")
test_raw  = pd.read_csv("raw_data/test.csv")
chief     = pd.read_csv("raw_data/chief_complaints.csv")
history   = pd.read_csv("raw_data/patient_history.csv")

print("Engineering features...", flush=True)
X, y, X_test, _ = build_features(train_raw, test_raw, history, chief)
print(f"  X: {X.shape}  X_test: {X_test.shape}", flush=True)

# ── Load tuned XGB params ─────────────────────────────────────────────────────
with open("submissions/best_params_xgb.json") as f:
    xgb_cfg = json.load(f)
print(f"\nTuned XGB (3-fold CV): {xgb_cfg['best_f1']:.4f}  n_est={xgb_cfg['best_n_estimators']}", flush=True)

# ── Model factories ───────────────────────────────────────────────────────────
def make_lgbm_default():
    """Untuned LGBM — best single model from prior runs (CV 0.9724)."""
    return lgb.LGBMClassifier(
        device="gpu", objective="multiclass", num_class=5,
        metric="multi_logloss", random_state=SEED, verbose=-1, n_jobs=-1,
        n_estimators=300, max_depth=7, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
    )

def make_xgb_tuned():
    return xgb.XGBClassifier(
        tree_method="hist", device="cuda", objective="multi:softmax",
        num_class=5, eval_metric="mlogloss", random_state=SEED,
        n_estimators=xgb_cfg["best_n_estimators"],
        **xgb_cfg["params"],
    )

# ── 5-fold OOF probabilities ──────────────────────────────────────────────────
print("\nRunning 5-fold CV for OOF probabilities...", flush=True)
n = len(y)
lgbm_oof = np.zeros((n, 5))
xgb_oof  = np.zeros((n, 5))
lgbm_f1s, xgb_f1s = [], []

for fold, (tr_idx, val_idx) in enumerate(SKF.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr  = y.iloc[tr_idx] - 1
    y_val = y.iloc[val_idx]

    m_lgbm = make_lgbm_default()
    m_lgbm.fit(X_tr, y_tr)
    lgbm_oof[val_idx] = m_lgbm.predict_proba(X_val)
    lgbm_f1s.append(f1_score(y_val, lgbm_oof[val_idx].argmax(1) + 1, average="macro", zero_division=0))

    m_xgb = make_xgb_tuned()
    m_xgb.fit(X_tr, y_tr)
    xgb_oof[val_idx] = m_xgb.predict_proba(X_val)
    xgb_f1s.append(f1_score(y_val, xgb_oof[val_idx].argmax(1) + 1, average="macro", zero_division=0))

    print(f"  Fold {fold+1}  LGBM={lgbm_f1s[-1]:.4f}  XGB={xgb_f1s[-1]:.4f}", flush=True)

lgbm_cv = np.mean(lgbm_f1s)
xgb_cv  = np.mean(xgb_f1s)
print(f"\n5-fold CV -> LGBM (default): {lgbm_cv:.4f}  XGB (tuned): {xgb_cv:.4f}", flush=True)

# ── Grid-search blend weight ──────────────────────────────────────────────────
print("\nOptimising blend weight alpha (LGBM*alpha + XGB*(1-alpha))...", flush=True)
best_alpha, best_blend_f1 = 0.0, 0.0
for alpha in np.arange(0.0, 1.01, 0.05):
    blended = alpha * lgbm_oof + (1 - alpha) * xgb_oof
    f1 = f1_score(y, blended.argmax(1) + 1, average="macro", zero_division=0)
    print(f"  alpha={alpha:.2f}  F1={f1:.4f}", flush=True)
    if f1 > best_blend_f1:
        best_blend_f1, best_alpha = f1, alpha

print(f"\nBest: alpha={best_alpha:.2f}  OOF F1={best_blend_f1:.4f}", flush=True)

# ── Strategy summary ──────────────────────────────────────────────────────────
strategies = {"lgbm_default": lgbm_cv, "xgb_tuned": xgb_cv, "blend": best_blend_f1}
best_strategy = max(strategies, key=strategies.get)
print(f"\nStrategy comparison:", flush=True)
for name, score in sorted(strategies.items(), key=lambda x: -x[1]):
    marker = " <-- best" if name == best_strategy else ""
    print(f"  {name:<15} {score:.4f}{marker}", flush=True)

# ── Retrain on full data ──────────────────────────────────────────────────────
print(f"\nRetraining on full data...", flush=True)
final_lgbm = make_lgbm_default()
final_lgbm.fit(X, y - 1)
lgbm_test = final_lgbm.predict_proba(X_test)

final_xgb = make_xgb_tuned()
final_xgb.fit(X, y - 1)
xgb_test = final_xgb.predict_proba(X_test)

# ── Generate predictions ──────────────────────────────────────────────────────
if best_strategy == "blend":
    test_proba = best_alpha * lgbm_test + (1 - best_alpha) * xgb_test
    test_preds = test_proba.argmax(1) + 1
    label = f"lgbm_xgb_a{best_alpha:.2f}"
elif best_strategy == "lgbm_default":
    test_preds = lgbm_test.argmax(1) + 1
    label = "lgbm_default"
else:
    test_preds = xgb_test.argmax(1) + 1
    label = "xgb_tuned"

# ── Save submission ───────────────────────────────────────────────────────────
best_f1  = strategies[best_strategy]
out_path = f"submissions/ensemble_{label}_{best_f1:.4f}.csv"
sub = pd.DataFrame({"patient_id": test_raw["patient_id"], "triage_acuity": test_preds})
sub.to_csv(out_path, index=False)
print(f"\nSubmission saved: {out_path}  ({len(sub):,} rows)", flush=True)
print(f"Best OOF macro F1: {best_f1:.4f}  (strategy: {best_strategy})", flush=True)

summary = {
    "best_strategy": best_strategy,
    "best_oof_f1": best_f1,
    "blend_alpha": float(best_alpha),
    "lgbm_default_5fold_cv": lgbm_cv,
    "xgb_tuned_5fold_cv": xgb_cv,
    "submission_file": out_path,
}
with open("submissions/ensemble_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved: submissions/ensemble_summary.json", flush=True)
