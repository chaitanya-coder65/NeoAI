import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from config.paths_config import (
    DATA_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    create_project_directories,
)


class ModelEvaluation:
    """
    Evaluate the trained baseline model on the held-out test dataset.
    The test set must not be used for model training or model selection.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path

    def load_test_data(self):
        X_test_path = DATA_DIR / "X_test_processed.csv"
        y_test_path = DATA_DIR / "y_test_processed.csv"

        if not X_test_path.exists() or not y_test_path.exists():
            raise FileNotFoundError(
                "Processed test data not found. Run Steps 3 and 4 first."
            )

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")

        if X_test.empty or y_test.empty:
            raise ValueError("Test dataset is empty.")

        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test have different row counts.")

        return X_test, y_test

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}. "
                "Run Step 5 or Step 6 first."
            )

        model = joblib.load(self.model_path)
        print("Trained model loaded:", self.model_path)
        return model

    def generate_predictions(self, model, X_test):
        y_pred = model.predict(X_test)

        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "The selected model does not support predict_proba; "
                "ROC-AUC requires probability scores or decision scores."
            )

        y_score = model.predict_proba(X_test)[:, 1]
        return y_pred, y_score

    def calculate_metrics(self, y_test, y_pred, y_score):
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(y_test, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, y_pred, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
        }

        return metrics

    def save_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # Save the numeric matrix as CSV.
        cm_df = pd.DataFrame(
            cm,
            index=["Actual_Malignant", "Actual_Benign"],
            columns=["Predicted_Malignant", "Predicted_Benign"],
        )
        cm_df.to_csv(REPORTS_DIR / "confusion_matrix.csv", index=True)

        # Save a readable visualization.
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Malignant (0)", "Benign (1)"],
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title("Baseline Model Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            REPORTS_DIR / "confusion_matrix.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        return cm

    def save_predictions(self, y_test, y_pred, y_score):
        predictions_df = pd.DataFrame({
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(y_pred),
            "predicted_probability_benign": pd.Series(y_score),
        })
        predictions_df.to_csv(
            REPORTS_DIR / "baseline_predictions.csv",
            index=False,
        )

    def save_evaluation_report(self, metrics, model, y_test, y_pred, cm):
        report = {
            "evaluation_type": "baseline_holdout_test_evaluation",
            "model_name": type(model).__name__,
            "test_samples": int(len(y_test)),
            "positive_class": "benign (1)",
            "metrics": metrics,
            "confusion_matrix": {
                "labels": ["malignant (0)", "benign (1)"],
                "values": cm.tolist(),
            },
        }

        report_path = REPORTS_DIR / "baseline_evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        # Human-readable report.
        text_report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            zero_division=0,
        )
        (REPORTS_DIR / "baseline_classification_report.txt").write_text(
            text_report,
            encoding="utf-8",
        )

        return report_path

    def initiate_model_evaluation(self):
        create_project_directories()

        X_test, y_test = self.load_test_data()
        model = self.load_model()

        y_pred, y_score = self.generate_predictions(model, X_test)
        metrics = self.calculate_metrics(y_test, y_pred, y_score)
        cm = self.save_confusion_matrix(y_test, y_pred)

        self.save_predictions(y_test, y_pred, y_score)
        report_path = self.save_evaluation_report(
            metrics, model, y_test, y_pred, cm
        )

        print("\nBaseline Model Evaluation Results")
        print("--------------------------------")
        print("Model:", type(model).__name__)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            zero_division=0,
        ))

        print("Evaluation report saved:", report_path)
        print("Evaluation completed successfully.")

        return metrics, cm


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.initiate_model_evaluation()
