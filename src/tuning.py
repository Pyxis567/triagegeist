"""Optuna hyperparameter tuning for LightGBM and XGBoost (3-fold CV, GPU).

Changes from v1:
- SQLite persistence: best params survive a kill/crash; study is resumable
- 3-fold CV (down from 5) for ~40% faster trials
- n_estimators capped at 2000 (was 3000) — early stopping handles the rest
- All prints flushed immediately
"""
import json, time, numpy as np, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb

from src.utils import SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_SPLITS = 3


def _make_skf():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# ── LightGBM objective ────────────────────────────────────────────────────────

def lgbm_objective(trial, X, y):
    params = dict(
        device="gpu",
        objective="multiclass",
        num_class=5,
        metric="multi_logloss",
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
        n_estimators=2000,
        num_leaves=trial.suggest_int("num_leaves", 31, 1000),
        max_depth=trial.suggest_int("max_depth", 4, 15),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 300),
        subsample=trial.suggest_float("subsample", 0.4, 1.0),
        subsample_freq=trial.suggest_int("subsample_freq", 1, 7),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        path_smooth=trial.suggest_float("path_smooth", 0.0, 1.0),
    )

    skf = _make_skf()
    fold_f1s, best_iters = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr  = y.iloc[tr_idx] - 1
        y_val = y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val - 1)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val) + 1
        fold_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_f1s.append(fold_f1)
        best_iters.append(model.best_iteration_ or params["n_estimators"])

        trial.report(float(np.mean(fold_f1s)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("best_n_estimators", int(np.mean(best_iters)))
    return float(np.mean(fold_f1s))


# ── XGBoost objective ─────────────────────────────────────────────────────────

def xgb_objective(trial, X, y):
    params = dict(
        tree_method="hist",
        device="cuda",
        objective="multi:softmax",
        num_class=5,
        eval_metric="mlogloss",
        random_state=SEED,
        n_estimators=2000,
        early_stopping_rounds=50,
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 50),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.4, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        colsample_bynode=trial.suggest_float("colsample_bynode", 0.4, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        max_delta_step=trial.suggest_int("max_delta_step", 0, 10),
    )

    skf = _make_skf()
    fold_f1s, best_iters = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr  = y.iloc[tr_idx] - 1
        y_val = y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val - 1)],
            verbose=False,
        )
        preds = model.predict(X_val) + 1
        fold_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        fold_f1s.append(fold_f1)
        best_iters.append(model.best_iteration or params["n_estimators"])

        trial.report(float(np.mean(fold_f1s)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("best_n_estimators", int(np.mean(best_iters)))
    return float(np.mean(fold_f1s))


# ── Study runner ──────────────────────────────────────────────────────────────

def tune_model(model_name, X, y, n_trials, save_path, db_path=None):
    """Run (or resume) an Optuna study; best params written to save_path after every trial."""
    assert model_name in ("lgbm", "xgb")
    objective = lgbm_objective if model_name == "lgbm" else xgb_objective

    storage = f"sqlite:///{db_path}" if db_path else None

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner  = optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=N_SPLITS, reduction_factor=3
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"triagegeist_{model_name}",
        storage=storage,
        load_if_exists=True,
    )

    already_done = len(study.trials)
    if already_done:
        print(f"  Resuming from {already_done} completed trials", flush=True)

    t0 = time.time()

    def _save_best(study, trial):
        """Persist best params to JSON after every completed (non-pruned) trial."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        best = study.best_trial
        result = {
            "model": model_name,
            "best_f1": best.value,
            "best_n_estimators": best.user_attrs.get("best_n_estimators", 300),
            "params": best.params,
            "n_trials_completed": len(study.trials),
        }
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)

    def _log(study, trial):
        elapsed = (time.time() - t0) / 60
        best    = study.best_value if study.best_trial else float("nan")
        n_done  = already_done + trial.number + 1
        status  = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else f"F1={trial.value:.4f}"
        print(
            f"  [{model_name.upper()}] trial {n_done:>4}/{already_done + n_trials}"
            f"  {status:<18}  best={best:.4f}  elapsed={elapsed:.1f}min",
            flush=True,
        )

    study.optimize(
        lambda t: objective(t, X, y),
        n_trials=n_trials,
        callbacks=[_save_best, _log],
        gc_after_trial=True,
    )

    best = study.best_trial
    result = {
        "model": model_name,
        "best_f1": best.value,
        "best_n_estimators": best.user_attrs.get("best_n_estimators", 300),
        "params": best.params,
        "n_trials_completed": len(study.trials),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved best params -> {save_path}", flush=True)
    print(f"  Best F1:           {result['best_f1']:.4f}", flush=True)
    print(f"  Best n_estimators: {result['best_n_estimators']}", flush=True)
    return result
