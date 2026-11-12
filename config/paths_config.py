from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
NOTEBOOK_DIR = ROOT_DIR / "notebook"
TESTS_DIR = ROOT_DIR / "tests"

BASELINE_DATA_PATH = DATA_DIR / "baseline_data.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
NEW_BATCH_DATA_PATH = DATA_DIR / "new_batch.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.json"
MONITORING_REPORT_PATH = REPORTS_DIR / "monitoring_report.json"

def create_project_directories():
    for path in (DATA_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOK_DIR, TESTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
