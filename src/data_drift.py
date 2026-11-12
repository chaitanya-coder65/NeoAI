import json
import pandas as pd
from scipy.stats import ks_2samp

from config.paths_config import (
    BASELINE_DATA_PATH,
    NEW_BATCH_DATA_PATH,
    REPORTS_DIR,
    create_project_directories,
)


class DataDrift:

    def __init__(
        self,
        significance_level=0.05,
        drift_share_threshold=0.30,
        target_column="target",
    ):
        self.significance_level = significance_level
        self.drift_share_threshold = drift_share_threshold
        self.target_column = target_column

    # 1. Load baseline and new batch datasets
    def load_datasets(self):
        if not BASELINE_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Baseline data not found: {BASELINE_DATA_PATH}. Run Step 2 first."
            )

        if not NEW_BATCH_DATA_PATH.exists():
            raise FileNotFoundError(
                f"New batch not found: {NEW_BATCH_DATA_PATH}. Run Step 8 first."
            )

        baseline_df = pd.read_csv(BASELINE_DATA_PATH)
        new_batch_df = pd.read_csv(NEW_BATCH_DATA_PATH)

        if baseline_df.empty or new_batch_df.empty:
            raise ValueError("Baseline or new batch dataset is empty.")

        if self.target_column not in baseline_df.columns:
            raise ValueError(f"Target column '{self.target_column}' is missing from baseline.")
        if self.target_column not in new_batch_df.columns:
            raise ValueError(f"Target column '{self.target_column}' is missing from new batch.")

        return baseline_df, new_batch_df

    # 2. Identify common numerical feature columns
    def get_numerical_features(self, baseline_df, new_batch_df):
        baseline_features = set(
            baseline_df.select_dtypes(include="number").columns
        )
        batch_features = set(
            new_batch_df.select_dtypes(include="number").columns
        )

        # Exclude target: drift analysis is for input features only.
        features = sorted(
            (baseline_features & batch_features) - {self.target_column}
        )

        if not features:
            raise ValueError("No common numerical feature columns found.")

        return features

    # 3. Apply KS two-sample test to each feature
    def calculate_feature_drift(
        self,
        baseline_df,
        new_batch_df,
        numerical_features,
    ):
        feature_results = []

        for feature in numerical_features:
            baseline_values = baseline_df[feature].dropna()
            batch_values = new_batch_df[feature].dropna()

            if baseline_values.empty or batch_values.empty:
                feature_results.append({
                    "feature": feature,
                    "ks_statistic": None,
                    "p_value": None,
                    "drift_detected": False,
                    "status": "skipped_empty_values",
                    "baseline_missing": int(baseline_df[feature].isna().sum()),
                    "new_batch_missing": int(new_batch_df[feature].isna().sum()),
                })
                continue

            ks_statistic, p_value = ks_2samp(
                baseline_values,
                batch_values,
                alternative="two-sided",
                method="auto",
            )

            drift_detected = bool(p_value < self.significance_level)

            feature_results.append({
                "feature": feature,
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "drift_detected": drift_detected,
                "status": "drift_detected" if drift_detected else "no_drift_detected",
                "baseline_missing": int(baseline_df[feature].isna().sum()),
                "new_batch_missing": int(new_batch_df[feature].isna().sum()),
            })

        return feature_results

    # 4. Calculate overall drift percentage and status
    def calculate_overall_drift(self, feature_results):
        total_features = len(feature_results)

        # A skipped feature is not counted as drifted.
        drifted_features = sum(
            1 for result in feature_results
            if result["drift_detected"]
        )

        drift_percentage = (
            (drifted_features / total_features) * 100
            if total_features else 0.0
        )

        drift_detected = (
            drift_percentage >= self.drift_share_threshold * 100
        )

        return {
            "total_features": total_features,
            "drifted_features": drifted_features,
            "drift_percentage": round(drift_percentage, 2),
            "drift_detected": bool(drift_detected),
        }

    # 5. Save JSON drift report
    def save_drift_report(
        self,
        baseline_df,
        new_batch_df,
        feature_results,
        overall_results,
    ):
        create_project_directories()

        report = {
            "method": "Kolmogorov-Smirnov two-sample test",
            "significance_level": self.significance_level,
            "drift_share_threshold": self.drift_share_threshold,
            "baseline_dataset": str(BASELINE_DATA_PATH),
            "new_batch_dataset": str(NEW_BATCH_DATA_PATH),
            "baseline_rows": int(len(baseline_df)),
            "new_batch_rows": int(len(new_batch_df)),
            **overall_results,
            "feature_results": feature_results,
            "data_note": (
                "The new batch is simulated production data generated for "
                "this assignment. Drift was deliberately introduced."
            ),
        }

        report_path = REPORTS_DIR / "drift_report.json"

        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print("Drift report saved:", report_path)
        return report

    # 6. Complete drift detection pipeline
    def initiate_drift_detection(self):
        baseline_df, new_batch_df = self.load_datasets()

        numerical_features = self.get_numerical_features(
            baseline_df,
            new_batch_df,
        )

        feature_results = self.calculate_feature_drift(
            baseline_df,
            new_batch_df,
            numerical_features,
        )

        overall_results = self.calculate_overall_drift(
            feature_results,
        )

        report = self.save_drift_report(
            baseline_df,
            new_batch_df,
            feature_results,
            overall_results,
        )

        print("\nData Drift Detection Results")
        print("----------------------------")
        print("Total features:", report["total_features"])
        print("Drifted features:", report["drifted_features"])
        print("Drift percentage:", report["drift_percentage"], "%")
        print("Overall drift detected:", report["drift_detected"])

        print("\nFeature-level results:")
        for result in feature_results:
            if result["p_value"] is None:
                print(f'{result["feature"]}: skipped')
            else:
                print(
                    f'{result["feature"]}: '
                    f'p-value={result["p_value"]:.6g}, '
                    f'drift={result["drift_detected"]}'
                )

        print("\nData drift detection completed successfully.")
        return report


if __name__ == "__main__":
    detector = DataDrift(
        significance_level=0.05,
        drift_share_threshold=0.30,
        target_column="target",
    )
    detector.initiate_drift_detection()
