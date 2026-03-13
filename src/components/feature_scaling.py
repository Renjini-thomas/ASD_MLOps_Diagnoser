import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class FeatureScaling:

    def __init__(self):

        self.input_dir = Path("artifacts/features")
        self.output_dir = Path("artifacts/scaled_features")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler_path = self.output_dir / "scaler.pkl"

    # --------------------------------------

    def run(self):

        print("Loading aggregated features...")

        train_df = pd.read_csv(self.input_dir / "train_features.csv")
        test_df = pd.read_csv(self.input_dir / "test_features.csv")

        id_train = train_df["subject_id"]
        label_train = train_df["label"]

        id_test = test_df["subject_id"]
        label_test = test_df["label"]

        X_train = train_df.drop(["subject_id", "label"], axis=1)
        X_test = test_df.drop(["subject_id", "label"], axis=1)

        scaler = StandardScaler()

        print("Fitting scaler on TRAIN data...")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        joblib.dump(scaler, self.scaler_path)

        print("Scaler saved →", self.scaler_path)

        train_scaled_df = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns
        )

        train_scaled_df.insert(0, "label", label_train)
        train_scaled_df.insert(0, "subject_id", id_train)

        test_scaled_df = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns
        )

        test_scaled_df.insert(0, "label", label_test)
        test_scaled_df.insert(0, "subject_id", id_test)

        train_scaled_df.to_csv(
            self.output_dir / "train_features.csv",
            index=False
        )

        test_scaled_df.to_csv(
            self.output_dir / "test_features.csv",
            index=False
        )

        print("Feature scaling completed")