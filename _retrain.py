import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import SEED
from src.feature_engineering import build_features
from src.models import run_rf, run_xgb, run_lgbm, run_nn

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs("submissions", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("Loading data...")
train_raw = pd.read_csv("raw_data/train.csv")
test_raw  = pd.read_csv("raw_data/test.csv")
chief     = pd.read_csv("raw_data/chief_complaints.csv")
history   = pd.read_csv("raw_data/patient_history.csv")

print("Engineering features...")
X_train, y_train, X_test, fit_params = build_features(train_raw, test_raw, history, chief)
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

train_fe = X_train.copy()
train_fe["triage_acuity"] = y_train.values

results = {}
_fig_counter = [0]
_orig_show = plt.show
def _save_show():
    _fig_counter[0] += 1
    plt.savefig(f"figures/fig_{_fig_counter[0]:02d}.png", dpi=100, bbox_inches="tight")
    plt.close("all")
plt.show = _save_show

print("\n=== Random Forest ===")
rf, rf_f1 = run_rf(train_fe, max_depth=20, n_estimators=200, criterion="entropy",
                   min_samples_split=10, max_features=0.5)
results["Random Forest"] = rf_f1
print(f"  >> Macro F1: {rf_f1:.4f}")

print("\n=== XGBoost ===")
xgb_model, xgb_f1 = run_xgb(train_fe)
results["XGBoost"] = xgb_f1
print(f"  >> Macro F1: {xgb_f1:.4f}")

print("\n=== LightGBM ===")
lgbm_model, lgbm_f1 = run_lgbm(train_fe)
results["LightGBM"] = lgbm_f1
print(f"  >> Macro F1: {lgbm_f1:.4f}")

print("\n=== Neural Network (MLP) ===")
nn_model, nn_scaler, nn_f1 = run_nn(train_fe, epochs=50,
                                    hidden_dims=(256, 128, 64),
                                    batch_size=512, lr=1e-3)
results["MLP"] = nn_f1
print(f"  >> Macro F1: {nn_f1:.4f}")

plt.show = _orig_show

print()
print("=" * 50)
print("  MODEL COMPARISON — Macro F1")
print("  (previous best: LightGBM 0.9725)")
print("=" * 50)
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    delta = score - 0.9725
    arrow = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    print(f"  {name:<20} {score:.4f}  ({arrow} vs baseline)")
print("=" * 50)

best_name  = max(results, key=results.get)
best_score = results[best_name]
print(f"\nBest model: {best_name}  ({best_score:.4f})")

if best_name == "Random Forest":
    preds = rf.predict(X_test)
elif best_name == "XGBoost":
    preds = xgb_model.predict(X_test) + 1
elif best_name == "LightGBM":
    preds = lgbm_model.predict(X_test) + 1
else:
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_te_s = nn_scaler.transform(X_test.values).astype(np.float32)
    with torch.no_grad():
        preds = nn_model(torch.tensor(X_te_s).to(_dev)).argmax(dim=1).cpu().numpy() + 1

out = f"submissions/{best_name.lower().replace(' ', '_')}_{best_score:.4f}.csv"
sub = pd.DataFrame({"patient_id": test_raw["patient_id"], "triage_acuity": preds})
sub.to_csv(out, index=False)
print(f"Saved: {out}  ({len(sub):,} rows)")
