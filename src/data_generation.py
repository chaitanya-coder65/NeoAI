import pandas as pd
import numpy as np

from config.paths_config import (
    BASELINE_DATA_PATH,
    DATA_DIR,
    NEW_BATCH_DATA_PATH,
    create_project_directories,
)


class DataGeneration:
    """
    Generate a reproducible simulated production batch from the reference
    dataset. Selected feature distributions are shifted to create drift.

    IMPORTANT:
    This is synthetic/simulated production data, not real production data.
    Labels are retained from the source rows for offline performance evaluation.
    """

    def __init__(
        self,
        batch_size=200,
        random_state=101,
        drift_fraction=0.50,
    ):
        self.batch_size = batch_size
        self.random_state = random_state
        self.drift_fraction = drift_fraction
        self.target_column = "target"

        # Features whose distributions will be shifted.
        self.drift_features = [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
        ]

    def load_reference_data(self):
        if not BASELINE_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Reference dataset not found: {BASELINE_DATA_PATH}. "
                "Run Step 2 first."
            )

        reference_df = pd.read_csv(BASELINE_DATA_PATH)

        if reference_df.empty:
            raise ValueError("Reference dataset is empty.")

        if self.target_column not in reference_df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' is missing."
            )

        if reference_df[self.target_column].isnull().any():
            raise ValueError("Reference target contains missing values.")

        missing_features = [
            feature for feature in self.drift_features
            if feature not in reference_df.columns
        ]
        if missing_features:
            raise ValueError(
                f"Drift features not found in reference data: {missing_features}"
            )

        print("Reference dataset loaded:", reference_df.shape)
        return reference_df

    def create_batch_sample(self, reference_df):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        # Sampling with replacement allows a batch larger than the reference data.
        batch_df = reference_df.sample(
            n=self.batch_size,
            replace=self.batch_size > len(reference_df),
            random_state=self.random_state,
        ).reset_index(drop=True)

        print("Initial simulated batch shape:", batch_df.shape)
        return batch_df

    def introduce_feature_drift(self, batch_df):
        rng = np.random.default_rng(self.random_state)

        if not 0 <= self.drift_fraction <= 1:
            raise ValueError("drift_fraction must be between 0 and 1.")

        number_to_change = int(len(batch_df) * self.drift_fraction)

        for feature in self.drift_features:
            # Shift a reproducible subset upward by 20% of its reference
            # standard deviation. This creates controlled covariate drift.
            std_value = batch_df[feature].std()

            if pd.isna(std_value) or std_value == 0:
                continue

            selected_rows = rng.choice(
                batch_df.index.to_numpy(),
                size=number_to_change,
                replace=False,
            )

            batch_df.loc[selected_rows, feature] = (
                batch_df.loc[selected_rows, feature]
                + 0.20 * std_value
            )

        print("Simulated drift introduced in:", self.drift_features)
        return batch_df

    def validate_batch_schema(self, reference_df, batch_df):
        if list(reference_df.columns) != list(batch_df.columns):
            raise ValueError(
                "New batch schema does not match reference schema."
            )

        if batch_df[self.target_column].isnull().any():
            raise ValueError("New batch labels contain missing values.")

        if len(batch_df) != self.batch_size:
            raise ValueError("Generated batch size is incorrect.")

        print("Feature schema validated successfully.")
        print("Labels retained successfully.")

    def save_batch(self, batch_df):
        create_project_directories()

        batch_df.to_csv(
            NEW_BATCH_DATA_PATH,
            index=False,
        )

        print("Simulated production batch saved:", NEW_BATCH_DATA_PATH)

    def generate_new_batch(self):
        reference_df = self.load_reference_data()
        batch_df = self.create_batch_sample(reference_df)
        batch_df = self.introduce_feature_drift(batch_df)

        self.validate_batch_schema(reference_df, batch_df)
        self.save_batch(batch_df)

        print("\nNew batch generation completed successfully.")
        print("Batch shape:", batch_df.shape)
        print("Target distribution:")
        print(batch_df[self.target_column].value_counts().sort_index())

        return batch_df


if __name__ == "__main__":
    generator = DataGeneration(
        batch_size=200,
        random_state=101,
        drift_fraction=0.50,
    )
    generator.generate_new_batch()
