import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(train_path="train.csv", test_path="test.csv"):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    target = "triage_acuity"
    drop   = ["patient_id", "triage_nurse_id", target]

    y = train[target].values
    X_train_raw = train.drop(columns=drop, errors="ignore")
    X_test_raw  = test.drop(columns=[c for c in drop if c in test.columns], errors="ignore")

    # Align columns
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns)

    # Encode categoricals
    cat_cols = X_train_raw.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_raw[col], X_test_raw[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_raw[col] = le.transform(X_train_raw[col].astype(str))
        X_test_raw[col]  = le.transform(X_test_raw[col].astype(str))
        encoders[col] = le

    X_train = X_train_raw.fillna(-1).values.astype(np.float32)
    X_test  = X_test_raw.fillna(-1).values.astype(np.float32)

    # Encode target to 0-indexed
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y)

    return X_train, X_test, y, label_enc


# ── Models ────────────────────────────────────────────────────────────────────

def build_rf(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=random_state,
    )


def build_xgb(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svm(C=1.0, kernel="rbf", random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(C=C, kernel=kernel, class_weight="balanced",
                       probability=True, random_state=random_state)),
    ])


# ── Neural Network ────────────────────────────────────────────────────────────

class TriageMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, num_classes, epochs=40, batch_size=256,
              lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)

    X_t = torch.tensor(X_scaled).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = TriageMLP(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    return model, scaler


def predict_mlp(model, scaler, X, device=None, batch_size=512):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_scaled = scaler.transform(X).astype(np.float32)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            xb = torch.tensor(X_scaled[i:i+batch_size]).to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ── Evaluation ────────────────────────────────────────────────────────────────

def cv_score(model, X, y, cv=5, scoring="f1_weighted"):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    print(f"  {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def evaluate(name, y_true, y_pred):
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(classification_report(y_true, y_pred))
    return f1_score(y_true, y_pred, average="weighted")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    train, test = load_data()
    X_train, X_test, y, label_enc = preprocess(train, test)
    num_classes = len(np.unique(y))
    print(f"  Train: {X_train.shape}, classes: {num_classes}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest
    print("\n[RF] Cross-validation...")
    rf = build_rf()
    cv_score(rf, X_train, y)
    rf.fit(X_train, y)

    # XGBoost
    print("\n[XGB] Cross-validation...")
    xgb_clf = build_xgb()
    cv_score(xgb_clf, X_train, y)
    xgb_clf.fit(X_train, y, eval_set=[(X_train, y)], verbose=False)

    # SVM
    print("\n[SVM] Cross-validation (sample for speed)...")
    svm = build_svm()
    sample = min(10_000, len(X_train))
    idx = np.random.default_rng(42).choice(len(X_train), sample, replace=False)
    cv_score(svm, X_train[idx], y[idx])
    svm.fit(X_train, y)

    # MLP
    print("\n[MLP] Training...")
    mlp, mlp_scaler = train_mlp(X_train, y, num_classes)

    # Evaluate on training set (use hold-out in practice)
    for name, preds in [
        ("Random Forest", rf.predict(X_train)),
        ("XGBoost",       xgb_clf.predict(X_train)),
        ("SVM",           svm.predict(X_train)),
        ("MLP",           predict_mlp(mlp, mlp_scaler, X_train)),
    ]:
        evaluate(name, y, preds)

    # Submission from best model (XGB by default — swap as needed)
    sub = pd.read_csv("sample_submission.csv")
    sub["triage_acuity"] = label_enc.inverse_transform(xgb_clf.predict(X_test))
    sub.to_csv("submission.csv", index=False)
    print("\nsubmission.csv written.")
