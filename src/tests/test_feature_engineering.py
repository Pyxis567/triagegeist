import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.feature_engineering import build_features


@pytest.fixture(scope="module")
def raw_data():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "raw_data")
    train   = pd.read_csv(os.path.join(base, "train.csv"))
    test    = pd.read_csv(os.path.join(base, "test.csv"))
    history = pd.read_csv(os.path.join(base, "patient_history.csv"))
    chief   = pd.read_csv(os.path.join(base, "chief_complaints.csv"))
    return train, test, history, chief


@pytest.fixture(scope="module")
def processed(raw_data):
    train, test, history, chief = raw_data
    return build_features(train, test, history, chief)


def test_no_nans_in_X_train(processed):
    X_train, _, _, _ = processed
    assert X_train.isna().sum().sum() == 0, "X_train contains NaN values"


def test_no_nans_in_X_test(processed):
    _, _, X_test, _ = processed
    assert X_test.isna().sum().sum() == 0, "X_test contains NaN values"


def test_train_test_columns_match(processed):
    X_train, _, X_test, _ = processed
    assert list(X_train.columns) == list(X_test.columns), (
        "Train and test column sets do not match"
    )


def test_no_leakage_columns(processed):
    X_train, _, X_test, _ = processed
    leakage = {"disposition", "ed_los_hours", "triage_acuity"}
    train_leak = leakage & set(X_train.columns)
    test_leak  = leakage & set(X_test.columns)
    assert not train_leak, f"Leakage columns in X_train: {train_leak}"
    assert not test_leak,  f"Leakage columns in X_test: {test_leak}"


def test_patient_id_dropped(processed):
    X_train, _, X_test, _ = processed
    assert "patient_id" not in X_train.columns
    assert "patient_id" not in X_test.columns


def test_medians_saved_in_fit_params(processed):
    _, _, _, fit_params = processed
    assert "medians" in fit_params
    impute_cols = [
        "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
        "pulse_pressure", "shock_index", "respiratory_rate", "temperature_c",
    ]
    for col in impute_cols:
        assert col in fit_params["medians"].index, f"Median missing for {col}"


def test_tfidf_vectorizer_saved(processed):
    _, _, _, fit_params = processed
    assert "tfidf" in fit_params
    assert hasattr(fit_params["tfidf"], "transform")


def test_tfidf_columns_present(processed):
    X_train, _, X_test, _ = processed
    tfidf_cols = [c for c in X_train.columns if c.startswith("tfidf_")]
    assert len(tfidf_cols) == 100, f"Expected 100 TF-IDF cols, got {len(tfidf_cols)}"


def test_new_engineered_features_present(processed):
    X_train, _, _, _ = processed
    expected_new = [
        "pulse_pressure_ratio", "map_hr_product",
        "comorbidity_count", "ed_admission_ratio",
        "is_high_news2", "is_medium_news2", "age_news2",
    ]
    missing = [f for f in expected_new if f not in X_train.columns]
    assert not missing, f"New engineered features missing: {missing}"


def test_only_numeric_columns(processed):
    X_train, _, X_test, _ = processed
    non_num_train = X_train.select_dtypes(exclude="number").columns.tolist()
    non_num_test  = X_test.select_dtypes(exclude="number").columns.tolist()
    assert not non_num_train, f"Non-numeric cols in X_train: {non_num_train}"
    assert not non_num_test,  f"Non-numeric cols in X_test: {non_num_test}"
