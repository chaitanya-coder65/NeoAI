import joblib
import pandas as pd
import pytest

from config.paths_config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    DATA_DIR,
)


@pytest.mark.integration
def test_saved_model_can_predict_on_test_data():
    required = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        DATA_DIR / "X_test.csv",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Run Steps 3-5 before this integration test.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    transformed = preprocessor.transform(X_test)
    X_processed = pd.DataFrame(
        transformed,
        columns=getattr(preprocessor, "feature_names_in_", X_test.columns),
        index=X_test.index,
    )
    predictions = model.predict(X_processed)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})
