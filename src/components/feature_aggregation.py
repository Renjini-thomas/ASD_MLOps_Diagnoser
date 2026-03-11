import pandas as pd
from pathlib import Path


class FeatureAggregation:

    def __init__(self):

        self.texture_dir = Path("artifacts/features")
        self.morph_dir = Path("artifacts/morphometric_features")

        self.output_dir = Path("artifacts/aggregated_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # CLEAN SUBJECT ID
    # ----------------------------------

    def clean_subject_id(self, df):

        if "subject_id" not in df.columns:
            raise ValueError("subject_id column missing")

        df["subject_id"] = (
            df["subject_id"]
            .astype(str)
            .str.replace(".png", "", regex=False)
            .str.replace(".mgz", "", regex=False)
        )

        return df

    # ----------------------------------
    # MERGE FUNCTION
    # ----------------------------------

    def merge_split(self, split):

        print(f"\nProcessing {split} split...")

        texture_file = self.texture_dir / f"{split}_features.csv"
        morph_file = self.morph_dir / f"{split}_morph_features.csv"

        texture_df = pd.read_csv(texture_file)
        morph_df = pd.read_csv(morph_file)

        texture_df = self.clean_subject_id(texture_df)
        morph_df = self.clean_subject_id(morph_df)

        # drop label from morph to avoid duplication
        if "label" in morph_df.columns:
            morph_df = morph_df.drop(columns=["label"])

        merged_df = texture_df.merge(
            morph_df,
            on="subject_id",
            how="inner"
        )

        print("Merged shape:", merged_df.shape)

        output_file = self.output_dir / f"{split}_features.csv"
        merged_df.to_csv(output_file, index=False)

        print(f"{split} aggregated features saved → {output_file}")

    # ----------------------------------
    # RUN
    # ----------------------------------

    def run(self):

        self.merge_split("train")
        self.merge_split("test")

        print("\nFeature aggregation completed successfully")