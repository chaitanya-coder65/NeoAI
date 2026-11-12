import numpy as np
import pandas as pd
import pytest

from src.data_ingestion import DataIngestion


def test_split_features_and_target():
    df = pd.DataFrame({
        "feature_a": range(20),
        "feature_b": [value * 2 for value in range(20)],
        "target": [0, 1] * 10,
    })

    ingestion = DataIngestion(test_size=0.2, random_state=42)
    X, y = ingestion.split_features_target(df)

    assert list(X.columns) == ["feature_a", "feature_b"]
    assert "target" not in X.columns
    assert len(X) == len(y) == 20


def test_split_data_is_reproducible_and_stratified():
    df = pd.DataFrame({
        "feature_a": range(100),
        "feature_b": [value * 3 for value in range(100)],
        "target": [0] * 60 + [1] * 40,
    })
    ingestion = DataIngestion(test_size=0.2, random_state=42)
    X, y = ingestion.split_features_target(df)

    first = ingestion.split_data(X, y)
    second = ingestion.split_data(X, y)

    X_train_1, X_test_1, y_train_1, y_test_1 = first
    X_train_2, X_test_2, y_train_2, y_test_2 = second

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)

    assert len(X_train_1) == 80
    assert len(X_test_1) == 20
    assert abs(y_train_1.mean() - y.mean()) < 0.03
    assert abs(y_test_1.mean() - y.mean()) < 0.08
