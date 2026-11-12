import pytest

from src.data_drift import DataDrift


def test_overall_drift_calculation():
    detector = DataDrift(
        significance_level=0.05,
        drift_share_threshold=0.30,
        target_column="target",
    )

    results = [
        {"feature": "a", "drift_detected": True},
        {"feature": "b", "drift_detected": True},
        {"feature": "c", "drift_detected": False},
        {"feature": "d", "drift_detected": False},
        {"feature": "e", "drift_detected": False},
    ]

    output = detector.calculate_overall_drift(results)

    assert output["total_features"] == 5
    assert output["drifted_features"] == 2
    assert output["drift_percentage"] == 40.0
    assert output["drift_detected"] is True


def test_no_drift_below_overall_threshold():
    detector = DataDrift(drift_share_threshold=0.30)

    results = [
        {"feature": "a", "drift_detected": True},
        {"feature": "b", "drift_detected": False},
        {"feature": "c", "drift_detected": False},
        {"feature": "d", "drift_detected": False},
        {"feature": "e", "drift_detected": False},
    ]

    output = detector.calculate_overall_drift(results)

    assert output["drift_percentage"] == 20.0
    assert output["drift_detected"] is False


@pytest.mark.integration
def test_drift_pipeline_creates_report():
    from config.paths_config import BASELINE_DATA_PATH, NEW_BATCH_DATA_PATH, DRIFT_REPORT_PATH

    if not BASELINE_DATA_PATH.exists() or not NEW_BATCH_DATA_PATH.exists():
        pytest.skip("Run Steps 2 and 8 before this integration test.")

    report = DataDrift().initiate_drift_detection()

    assert report["total_features"] > 0
    assert len(report["feature_results"]) == report["total_features"]
    assert DRIFT_REPORT_PATH.exists()
