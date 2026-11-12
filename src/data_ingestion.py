import pandas as pd
from sklearn.model_selection import train_test_split

from config.paths_config import (
    BASELINE_DATA_PATH,
    DATA_DIR,
    create_project_directories,
)


class DataIngestion:

    def __init__(
        self,
        test_size=0.20,
        random_state=42,
        target_column="target"
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.target_column = target_column

    def load_data(self):
        if not BASELINE_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {BASELINE_DATA_PATH}. "
                "Run Step 2 first."
            )

        df = pd.read_csv(BASELINE_DATA_PATH)

        if df.empty:
            raise ValueError("Baseline dataset is empty.")

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found."
            )

        if df[self.target_column].isnull().any():
            raise ValueError("Target column contains missing values.")

        print("Dataset loaded successfully.")
        print("Dataset shape:", df.shape)
        return df

    def split_features_target(self, df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        print("Features shape:", X.shape)
        print("Target shape:", y.shape)
        return X, y

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        print("Training features:", X_train.shape)
        print("Testing features:", X_test.shape)
        print("Training target:", y_train.shape)
        print("Testing target:", y_test.shape)

        return X_train, X_test, y_train, y_test

    def save_data(self, X_train, X_test, y_train, y_test):
        create_project_directories()

        X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
        X_test.to_csv(DATA_DIR / "X_test.csv", index=False)

        y_train.to_frame(name=self.target_column).to_csv(
            DATA_DIR / "y_train.csv", index=False
        )
        y_test.to_frame(name=self.target_column).to_csv(
            DATA_DIR / "y_test.csv", index=False
        )

        print("All datasets saved successfully.")

    def initiate_data_ingestion(self):
        df = self.load_data()
        X, y = self.split_features_target(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.save_data(X_train, X_test, y_train, y_test)

        print("Data ingestion completed successfully.")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    ingestion = DataIngestion(
        test_size=0.20,
        random_state=42,
        target_column="target"
    )
    ingestion.initiate_data_ingestion()
