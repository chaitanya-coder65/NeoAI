import json
import pytest

from config.paths_config import MONITORING_REPORT_PATH


@pytest.mark.integration
def test_monitoring_report_has_expected_sections():
    if not MONITORING_REPORT_PATH.exists():
        pytest.skip("Run Step 10 before this integration test.")

    with open(MONITORING_REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    assert report["status"] in {"HEALTHY", "ALERT"}
    assert "alerts" in report
    assert "model_evaluation" in report
    assert "data_drift" in report
    assert "data_drift" in report["alerts"]
    assert "performance_degradation" in report["alerts"]
