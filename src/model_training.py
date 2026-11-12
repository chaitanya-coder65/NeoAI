import json
import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

from config.paths_config import DATA_DIR, REPORTS_DIR, MODEL_PATH, create_project_directories
from utils.mlflow_utils import MLflowTracker


class ModelTraining:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self.get_models()
        self.tracker = MLflowTracker()

    def get_models(self):
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=self.random_state,
                class_weight="balanced", n_jobs=-1
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=200, random_state=self.random_state,
                class_weight="balanced", n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=3,
                random_state=self.random_state
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.1, random_state=self.random_state
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=self.random_state
            ),
            "SVM": SVC(
                kernel="rbf", C=1.0, probability=True,
                class_weight="balanced", random_state=self.random_state
            ),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }

    def load_data(self):
        paths = {
            "X_train": DATA_DIR / "X_train_processed.csv",
            "X_test": DATA_DIR / "X_test_processed.csv",
            "y_train": DATA_DIR / "y_train_processed.csv",
            "y_test": DATA_DIR / "y_test_processed.csv",
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run Steps 3 and 4 first.")

        X_train = pd.read_csv(paths["X_train"])
        X_test = pd.read_csv(paths["X_test"])
        y_train = pd.read_csv(paths["y_train"]).squeeze("columns")
        y_test = pd.read_csv(paths["y_test"]).squeeze("columns")

        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError("Feature and target row counts do not match.")
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("Training and testing columns do not match.")

        print("Training data:", X_train.shape)
        print("Testing data:", X_test.shape)
        return X_train, X_test, y_train, y_test

    def compare_models(self, X_train, y_train):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        results = []

        print("\nRunning 5-fold cross-validation and logging MLflow runs...")

        for name, model in self.models.items():
            scores = cross_validate(
                model, X_train, y_train, cv=cv,
                scoring=scoring, n_jobs=-1, error_score="raise"
            )
            metrics = {
                "cv_accuracy": scores["test_accuracy"].mean(),
                "cv_precision": scores["test_precision"].mean(),
                "cv_recall": scores["test_recall"].mean(),
                "cv_f1": scores["test_f1"].mean(),
                "cv_roc_auc": scores["test_roc_auc"].mean(),
            }

            self.tracker.log_model_run(
                run_name=f"{name}_cross_validation",
                model_name=name,
                params=model.get_params(),
                metrics=metrics,
                tags={"stage": "cross_validation", "selection_metric": "cv_f1"},
            )
            results.append({"model": name, **metrics})
            print(f"{name}: CV F1 = {metrics['cv_f1']:.4f}")

        results_df = pd.DataFrame(results).sort_values(
            "cv_f1", ascending=False
        ).reset_index(drop=True)

        create_project_directories()
        results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        return results_df

    def train_best_model(self, results_df, X_train, y_train):
        best_name = results_df.iloc[0]["model"]
        best_model = self.models[best_name]

        print("\nSelected model:", best_name)
        best_model.fit(X_train, y_train)

        create_project_directories()
        joblib.dump(best_model, MODEL_PATH)
        print("Saved model:", MODEL_PATH)
        return best_model, best_name

    def evaluate_model(self, model, model_name, X_test, y_test):
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        print("\nHoldout test metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test, predictions,
            target_names=["malignant", "benign"], zero_division=0
        ))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))

        create_project_directories()
        (REPORTS_DIR / "model_metrics.json").write_text(
            json.dumps({"selected_model": model_name, **metrics}, indent=4),
            encoding="utf-8"
        )
        pd.DataFrame({
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(predictions)
        }).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

        # The selected fitted model and its final metrics are stored in MLflow.
        self.tracker.log_model_run(
            run_name=f"{model_name}_final_selected_model",
            model_name=model_name,
            params=model.get_params(),
            metrics=metrics,
            model=model,
            log_model=True,
            tags={"stage": "final_evaluation", "selected_model": "true"},
        )
        return metrics, predictions

    def initiate_model_training(self):
        X_train, X_test, y_train, y_test = self.load_data()
        results = self.compare_models(X_train, y_train)
        model, model_name = self.train_best_model(results, X_train, y_train)
        metrics, predictions = self.evaluate_model(model, model_name, X_test, y_test)
        print("\nTraining and MLflow tracking completed.")
        return model, metrics, predictions


if __name__ == "__main__":
    ModelTraining(random_state=42).initiate_model_training()
