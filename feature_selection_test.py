"""
Quick feature-selection experiment for LightGBM.

Tests three pruning strategies against the full-feature baseline:
  1. Drop zero-importance features (importance == 0 in every fold)
  2. Drop low-importance features (cumulative importance threshold)
  3. Drop highly-correlated redundant features (|r| > 0.95)

All strategies are evaluated with the same 5-fold CV so results are
directly comparable to the baseline CV F1 of 0.9724.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from src.feature_engineering import build_features
from src.utils import SEED

np.random.seed(SEED)

# ── Load & engineer features ──────────────────────────────────────────────────
print("Loading data and engineering features...")
train_raw = pd.read_csv("raw_data/train.csv")
test_raw  = pd.read_csv("raw_data/test.csv")
chief     = pd.read_csv("raw_data/chief_complaints.csv")
history   = pd.read_csv("raw_data/patient_history.csv")
X_train, y_train, X_test, _ = build_features(train_raw, test_raw, history, chief)
print(f"  Full feature set: {X_train.shape[1]} features")

# ── CV helper ─────────────────────────────────────────────────────────────────
device = "gpu" if torch.cuda.is_available() else "cpu"
LGB_PARAMS = dict(
    device=device, objective="multiclass", num_class=5,
    metric="multi_logloss", random_state=SEED,
    n_estimators=300, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
)

def cv_score(X, y, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=int)
    imp = np.zeros(X.shape[1])
    for tr, val in skf.split(X, y):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr] - 1)
        oof[val] = m.predict(X.iloc[val]) + 1
        imp += m.feature_importances_
    f1 = f1_score(y, oof, average="macro", zero_division=0)
    print(f"  [{label}]  features={X.shape[1]:>3d}  OOF macro F1={f1:.4f}")
    return f1, imp / 5


# ── Baseline ─────────────────────────────────────────────────────────────────
print("\n--- Baseline (all features) ---")
base_f1, avg_imp = cv_score(X_train, y_train, "baseline")

imp_series = pd.Series(avg_imp, index=X_train.columns).sort_values(ascending=False)

# ── Strategy 1: Drop zero-importance features ─────────────────────────────────
print("\n--- Strategy 1: drop zero-importance features ---")
keep_nonzero = imp_series[imp_series > 0].index
X_s1 = X_train[keep_nonzero]
dropped_s1 = X_train.shape[1] - X_s1.shape[1]
print(f"  Dropping {dropped_s1} zero-importance features")
s1_f1, _ = cv_score(X_s1, y_train, "no-zero-imp")

# ── Strategy 2: Keep top-N by cumulative importance (95% threshold) ───────────
print("\n--- Strategy 2: keep features covering 95% of cumulative importance ---")
cum = imp_series.cumsum() / imp_series.sum()
keep_95 = cum[cum <= 0.95].index.tolist()
# include the feature that tips it over 95%
tip = cum[cum > 0.95].index[0] if (cum > 0.95).any() else []
keep_95 = keep_95 + ([tip] if len(tip) > 0 else [])
X_s2 = X_train[keep_95]
print(f"  Keeping {len(keep_95)} features (dropping {X_train.shape[1] - len(keep_95)})")
s2_f1, _ = cv_score(X_s2, y_train, "top-95pct-imp")

# ── Strategy 3: Drop highly-correlated redundant features ─────────────────────
print("\n--- Strategy 3: drop features with |r| > 0.95 to any higher-importance feature ---")
# Sort by importance descending; greedily keep, drop if corr > threshold with any kept col
corr_thresh = 0.95
sorted_cols = imp_series.index.tolist()
kept = []
corr_matrix = X_train[sorted_cols].corr().abs()
for col in sorted_cols:
    if not kept:
        kept.append(col)
        continue
    max_corr = corr_matrix.loc[col, kept].max()
    if max_corr < corr_thresh:
        kept.append(col)
X_s3 = X_train[kept]
print(f"  Keeping {len(kept)} features (dropping {X_train.shape[1] - len(kept)})")
s3_f1, _ = cv_score(X_s3, y_train, "dedup-corr95")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEATURE SELECTION SUMMARY  (LightGBM 5-fold CV)")
print("=" * 60)
rows = [
    ("Baseline",              X_train.shape[1], base_f1),
    ("Drop zero-importance",  X_s1.shape[1],    s1_f1),
    ("Top 95% cum-importance",len(keep_95),      s2_f1),
    ("Dedup corr > 0.95",     X_s3.shape[1],    s3_f1),
]
for name, n_feat, f1 in rows:
    delta = f1 - base_f1
    arrow = f"({delta:+.4f})" if name != "Baseline" else "         "
    print(f"  {name:<28}  {n_feat:>3d} feats  F1={f1:.4f}  {arrow}")
print("=" * 60)

# ── Show top-30 and bottom-20 by importance ───────────────────────────────────
print("\nTop 20 features by avg LightGBM importance:")
for col, val in imp_series.head(20).items():
    print(f"  {col:<45} {val:,.0f}")

print(f"\nZero-importance features ({dropped_s1}):")
for col in imp_series[imp_series == 0].index:
    print(f"  {col}")

os._exit(0)
