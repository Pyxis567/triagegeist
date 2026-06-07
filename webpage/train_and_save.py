"""
One-time script: trains LGBM + XGB on full training data and saves
models + fit_params to webpage/models/.

Run from the triagegeist root:
    python webpage/train_and_save.py

Requires: pip install flask joblib (in addition to the existing dsc80 env)
"""
import sys, os, json
_WEBPAGE = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_WEBPAGE)
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
import torch
import warnings
warnings.filterwarnings("ignore")

from src.utils import SEED
from src.feature_engineering import build_features

np.random.seed(SEED)

MODELS_DIR = os.path.join(_WEBPAGE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading data...", flush=True)
train_raw = pd.read_csv("raw_data/train.csv")
test_raw  = pd.read_csv("raw_data/test.csv")
chief     = pd.read_csv("raw_data/chief_complaints.csv")
history   = pd.read_csv("raw_data/patient_history.csv")

print("Engineering features...", flush=True)
X, y, _, fit_params = build_features(train_raw, test_raw, history, chief)
print(f"  X shape: {X.shape}", flush=True)

fit_params["hx_cols"]         = [c for c in history.columns if c != "patient_id"]
fit_params["feature_columns"] = X.columns.tolist()

joblib.dump(fit_params, os.path.join(MODELS_DIR, "fit_params.pkl"))
print("Saved fit_params.pkl", flush=True)

with open("submissions/best_params_xgb.json") as f:
    xgb_cfg = json.load(f)

device_lgbm = "gpu" if torch.cuda.is_available() else "cpu"
device_xgb  = "cuda" if torch.cuda.is_available() else "cpu"

print("\nTraining LightGBM...", flush=True)
lgbm_model = lgb.LGBMClassifier(
    device=device_lgbm, objective="multiclass", num_class=5,
    metric="multi_logloss", random_state=SEED, verbose=-1, n_jobs=-1,
    n_estimators=300, max_depth=7, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
)
lgbm_model.fit(X, y - 1)
joblib.dump(lgbm_model, os.path.join(MODELS_DIR, "lgbm_model.pkl"))
print("Saved lgbm_model.pkl", flush=True)

print("\nTraining XGBoost...", flush=True)
xgb_model = xgb.XGBClassifier(
    tree_method="hist", device=device_xgb,
    objective="multi:softmax", num_class=5, eval_metric="mlogloss",
    random_state=SEED, n_estimators=xgb_cfg["best_n_estimators"],
    **xgb_cfg["params"],
)
xgb_model.fit(X, y - 1)
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
print("Saved xgb_model.pkl", flush=True)

print("\nDone — all models saved to webpage/models/", flush=True)
