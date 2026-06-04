import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.utils import SEED, evaluate, plot_confusion_matrix, plot_feature_importance, plot_f1_by_class

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split(df, target, test_size):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def _plot_results(y_test, preds, feature_names=None, importances=None, title="Model"):
    classes = sorted(y_test.unique())
    report  = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc, macro_f1 = evaluate(y_test, preds)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, classes, axes[0])

    # Feature importances (if available)
    if importances is not None and feature_names is not None:
        imp_series = pd.Series(importances, index=feature_names)
        plot_feature_importance(imp_series, axes[1])
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Feature importance\nnot available",
                     ha="center", va="center", fontsize=12)

    # Per-class F1
    plot_f1_by_class(report, classes, macro_f1, axes[2])

    plt.tight_layout()
    plt.show()
    return macro_f1


# ── XGBoost (5-fold CV) ───────────────────────────────────────────────────────

def run_xgb(df, target="triage_acuity", n_splits=5, **xgb_params):
    X = df.drop(columns=[target])
    y = df[target]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    defaults = dict(
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        num_class=5,
        objective="multi:softmax",
        random_state=SEED,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    defaults.update(xgb_params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds      = np.zeros(len(y), dtype=int)
    importances_sum = np.zeros(X.shape[1])
    fold_f1s       = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        model = xgb.XGBClassifier(**defaults)
        model.fit(X_tr, y.iloc[tr_idx] - 1)
        fold_preds = model.predict(X_val) + 1
        oof_preds[val_idx] = fold_preds
        importances_sum += model.feature_importances_
        fold_f1 = f1_score(y_val, fold_preds, average="macro", zero_division=0)
        fold_f1s.append(fold_f1)
        print(f"  Fold {fold + 1}/{n_splits}  F1={fold_f1:.4f}")

    print(f"  CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    macro_f1 = _plot_results(y, oof_preds,
                              feature_names=X.columns,
                              importances=importances_sum / n_splits,
                              title=f"XGBoost ({n_splits}-fold CV)")

    # Retrain on full data for test-set predictions
    final = xgb.XGBClassifier(**defaults)
    final.fit(X, y - 1)
    return final, macro_f1


# ── LightGBM (5-fold CV) ──────────────────────────────────────────────────────

def run_lgbm(df, target="triage_acuity", n_splits=5, **lgbm_params):
    X = df.drop(columns=[target])
    y = df[target]

    device = "gpu" if torch.cuda.is_available() else "cpu"
    defaults = dict(
        device=device,
        objective="multiclass",
        num_class=5,
        metric="multi_logloss",
        random_state=SEED,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(lgbm_params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds      = np.zeros(len(y), dtype=int)
    importances_sum = np.zeros(X.shape[1])
    fold_f1s       = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        model = lgb.LGBMClassifier(**defaults)
        model.fit(X_tr, y.iloc[tr_idx] - 1)
        fold_preds = model.predict(X_val) + 1
        oof_preds[val_idx] = fold_preds
        importances_sum += model.feature_importances_
        fold_f1 = f1_score(y_val, fold_preds, average="macro", zero_division=0)
        fold_f1s.append(fold_f1)
        print(f"  Fold {fold + 1}/{n_splits}  F1={fold_f1:.4f}")

    print(f"  CV Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    macro_f1 = _plot_results(y, oof_preds,
                              feature_names=X.columns,
                              importances=importances_sum / n_splits,
                              title=f"LightGBM ({n_splits}-fold CV)")

    # Retrain on full data for test-set predictions
    final = lgb.LGBMClassifier(**defaults)
    final.fit(X, y - 1)
    return final, macro_f1


# ── Neural Network (MLP) ─────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_nn(df, target="triage_acuity", test_size=0.2,
           epochs=50, hidden_dims=(256, 128, 64), batch_size=512, lr=1e-3):
    X_train, X_test, y_train, y_test = _split(df, target, test_size)

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.values).astype(np.float32)
    X_te = scaler.transform(X_test.values).astype(np.float32)

    # 0-indexed labels
    y_tr = (y_train.values - 1).astype(np.int64)
    y_te = (y_test.values  - 1).astype(np.int64)

    # Tensors → device
    tr_ds = TensorDataset(torch.tensor(X_tr).to(_DEVICE),
                          torch.tensor(y_tr).to(_DEVICE))
    te_ds = TensorDataset(torch.tensor(X_te).to(_DEVICE),
                          torch.tensor(y_te).to(_DEVICE))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size)

    model = _MLP(X_tr.shape[1], hidden_dims, n_classes=5).to(_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(tr_ds))
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={train_losses[-1]:.4f}")

    # Evaluate
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in te_loader:
            all_preds.append(model(xb).argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds) + 1  # shift back to 1-indexed
    y_te_orig = y_test  # original 1-indexed

    macro_f1 = _plot_results(y_te_orig, preds, title="MLP (PyTorch)")

    # Training loss curve (4th plot)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs + 1), train_losses, color="#4C72B0", linewidth=1.5)
    ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    plt.tight_layout()
    plt.show()

    return model, scaler, macro_f1
