import pandas as pd
from pathlib import Path


class FeatureAggregation:

    def __init__(self):

        self.texture_dir = Path("artifacts/features")
        self.morph_dir = Path("artifacts/morphometric_features")

        self.output_dir = Path("artifacts/aggregated_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # NORMALIZE SUBJECT ID COLUMN
    # ------------------------------------------------

    def normalize_subject_id(self, df):

        possible_cols = ["subject_id", "image_id", "filename", "image", "file", "path"]

        found = None

        for col in possible_cols:
            if col in df.columns:
                found = col
                break

        if found is None:
            raise ValueError(
                f"No subject identifier column found. Columns = {df.columns.tolist()}"
            )

        df = df.rename(columns={found: "subject_id"})

        df["subject_id"] = (
            df["subject_id"]
            .astype(str)
            .str.replace(".png", "", regex=False)
            .str.replace(".mgz", "", regex=False)
            .str.strip()
        )

        return df

    # ------------------------------------------------
    # MERGE SPLIT
    # ------------------------------------------------

    def merge_split(self, split):

        print(f"\nProcessing {split} split...")

        texture_file = self.texture_dir / f"{split}_features.csv"
        morph_file = self.morph_dir / f"{split}_morph_features.csv"

        texture_df = pd.read_csv(texture_file)
        morph_df = pd.read_csv(morph_file)

        texture_df = self.normalize_subject_id(texture_df)
        morph_df = self.normalize_subject_id(morph_df)

        # Avoid duplicate label column
        if "label" in morph_df.columns:
            morph_df = morph_df.drop(columns=["label"])

        print("Texture shape:", texture_df.shape)
        print("Morph shape:", morph_df.shape)

        merged_df = texture_df.merge(
            morph_df,
            on="subject_id",
            how="inner"
        )

        print("Merged shape:", merged_df.shape)

        if merged_df.shape[0] == 0:
            raise ValueError("Fusion produced EMPTY dataset → subject_id mismatch")

        output_file = self.output_dir / f"{split}_features.csv"
        merged_df.to_csv(output_file, index=False)

        print(f"{split} aggregated features saved → {output_file}")

    # ------------------------------------------------
    # RUN
    # ------------------------------------------------

    def run(self):

        self.merge_split("train")
        self.merge_split("test")

        print("\nFeature aggregation completed successfully")