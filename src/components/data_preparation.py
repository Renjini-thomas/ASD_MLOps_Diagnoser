import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataPreparation:

    def __init__(self):

        self.pheno_file = Path("data/raw/phenotypic.csv")
        self.mri_dir = Path("data/raw/brain_mgz")

        self.interim_dir = Path("data/interim")
        self.processed_dir = Path("data/processed")

    # -----------------------------
    # CREATE MANIFEST
    # -----------------------------

    def create_manifest(self):

        df = pd.read_csv(self.pheno_file)

        dataset = []

        for subject_dir in self.mri_dir.iterdir():

            subject_id = subject_dir.name
            mri_path = subject_dir / "mri" / "brain.mgz"

            if not mri_path.exists():
                continue

            sub_id = int(subject_id.split("_")[-1])

            row = df[df["SUB_ID"] == sub_id]

            if row.empty:
                continue

            dx = row.iloc[0]["DX_GROUP"]

            label = "autism" if dx == 1 else "control"

            dataset.append({
                "subject_id": subject_id,
                "label": label,
                "subject_path": str(subject_dir)
            })

        manifest = pd.DataFrame(dataset)

        manifest_path = self.interim_dir / "dataset_manifest.csv"
        manifest.to_csv(manifest_path, index=False)

        print("Manifest created:", manifest_path)

        return manifest

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------

    def split_dataset(self, manifest):

        train, test = train_test_split(
            manifest,
            test_size=0.2,
            stratify=manifest["label"],
            random_state=42
        )

        train_path = self.interim_dir / "train_manifest.csv"
        test_path = self.interim_dir / "test_manifest.csv"

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        return train, test

    # -----------------------------
    # SAVE DATASET
    # -----------------------------

    def save_dataset(self, df, split):

        for _, row in df.iterrows():

            label = row["label"]
            src = Path(row["subject_path"])

            dest_dir = self.processed_dir / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest = dest_dir / row["subject_id"]

            if dest.exists():
                continue

            shutil.copytree(src, dest)

    # -----------------------------
    # RUN STAGE
    # -----------------------------

    def run(self):

        # VERY IMPORTANT FOR DVC
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.create_manifest()

        train, test = self.split_dataset(manifest)

        self.save_dataset(train, "train")
        self.save_dataset(test, "test")

        print("Data preparation completed")