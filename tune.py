"""Entry point for hyperparameter tuning.

Usage:
    python tune.py --model lgbm --n-trials 30
    python tune.py --model xgb  --n-trials 50
    python tune.py --model lgbm --n-trials 5   # GPU smoke test

Re-running the same --model will resume from the SQLite DB automatically.
"""
import sys, os, argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from src.utils import SEED
from src.feature_engineering import build_features
from src.tuning import tune_model

np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--model",    choices=["lgbm", "xgb"], required=True)
parser.add_argument("--n-trials", type=int, default=10)
args = parser.parse_args()

os.makedirs("submissions", exist_ok=True)

print("Loading data...", flush=True)
train_raw = pd.read_csv("raw_data/train.csv")
test_raw  = pd.read_csv("raw_data/test.csv")
chief     = pd.read_csv("raw_data/chief_complaints.csv")
history   = pd.read_csv("raw_data/patient_history.csv")

print("Engineering features...", flush=True)
X_train, y_train, X_test, _ = build_features(train_raw, test_raw, history, chief)
print(f"  X_train: {X_train.shape}", flush=True)

save_path = f"submissions/best_params_{args.model}.json"
db_path   = f"submissions/optuna_{args.model}.db"

print(f"\nStarting {args.model.upper()} tuning — {args.n_trials} trials (3-fold CV, GPU)\n", flush=True)

result = tune_model(
    args.model, X_train, y_train,
    n_trials=args.n_trials,
    save_path=save_path,
    db_path=db_path,
)

print(f"\n{'='*50}", flush=True)
print(f"  {args.model.upper()} tuning complete", flush=True)
print(f"  Best CV macro F1 : {result['best_f1']:.4f}", flush=True)
print(f"  Best n_estimators: {result['best_n_estimators']}", flush=True)
print(f"  Elapsed          : {result['elapsed_min']} min", flush=True)
print(f"  Params saved to  : {save_path}", flush=True)
print(f"{'='*50}", flush=True)
