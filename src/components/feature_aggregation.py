import pandas as pd
from pathlib import Path


class FeatureAggregation:

    def __init__(self):

        self.input_dir = Path("artifacts/features")
        self.output_dir = Path("artifacts/aggregated_features")

        self.output_dir.mkdir(parents=True, exist_ok=True)


    def aggregate_split(self, split):

        df = pd.read_csv(self.input_dir / f"{split}_features.csv")

        # Extract subject id
        df["subject"] = df["image_id"].str.replace(r"_s-?\d+$", "", regex=True)

        feature_cols = [c for c in df.columns if c not in ["image_id", "label", "subject"]]

        # Mean aggregation
        mean_features = df.groupby("subject")[feature_cols].mean()

        # Std aggregation
        std_features = df.groupby("subject")[feature_cols].std()

        std_features = std_features.add_suffix("_std")
        mean_features = mean_features.add_suffix("_mean")

        aggregated = pd.concat([mean_features, std_features], axis=1)

        aggregated["label"] = df.groupby("subject")["label"].first()

        aggregated.reset_index(inplace=True)

        aggregated.to_csv(self.output_dir / f"{split}_aggregated.csv", index=False)


    def run(self):

        self.aggregate_split("train")
        self.aggregate_split("test")