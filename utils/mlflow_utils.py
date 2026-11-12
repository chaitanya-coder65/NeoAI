import mlflow
import mlflow.sklearn

from config.paths_config import ROOT_DIR
from config.config import load_config


class MLflowTracker:
    """Configure MLflow and log model-training runs."""

    def __init__(self):
        config = load_config()
        tracking_uri = config["mlflow"]["tracking_uri"]
        experiment_name = config["mlflow"]["experiment_name"]

        # Resolve sqlite:///mlflow.db relative to the project root.
        if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
            db_name = tracking_uri.replace("sqlite:///", "", 1)
            tracking_uri = f"sqlite:///{(ROOT_DIR / db_name).resolve().as_posix()}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

        experiment = mlflow.get_experiment_by_name(experiment_name)
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        print("MLflow experiment:", experiment_name)
        print("MLflow experiment ID:", experiment.experiment_id)

    def log_model_run(
        self, run_name, model_name, params, metrics,
        model=None, log_model=False, tags=None
    ):
        """Log parameters, metrics, tags and an optional fitted model."""
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("model_name", model_name)

            if params:
                mlflow.log_params({
                    str(key): str(value)
                    for key, value in params.items()
                    if value is not None
                })

            if metrics:
                mlflow.log_metrics({
                    str(key): float(value)
                    for key, value in metrics.items()
                    if value is not None
                })

            if tags:
                mlflow.set_tags(tags)

            if log_model and model is not None:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )

            print("Logged MLflow run:", run.info.run_id)
            return run.info.run_id
