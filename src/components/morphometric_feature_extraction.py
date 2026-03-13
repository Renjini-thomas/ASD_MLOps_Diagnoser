import pandas as pd
from pathlib import Path


class MorphometricFeatureExtraction:

    def __init__(self):

        self.input_dir = Path("data/processed")
        self.output_dir = Path("artifacts/morphometric_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------
    # PARSE ASEG FULL TABLE
    # --------------------------------

    def parse_aseg(self, file):

        features = {}

        with open(file) as f:
            for line in f:

                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                tokens = line.split()

                # Only table rows start with numeric index
                if not tokens[0].isdigit():
                    continue

                if len(tokens) < 5:
                    continue

                region = tokens[4]
                volume = float(tokens[3])

                # clean feature name
                feature_name = (
                    region.lower()
                    .replace("-", "_")
                    .replace(".", "")
                )

                features[f"vol_{feature_name}"] = volume

        return features

    # --------------------------------
    # PROCESS SPLIT
    # --------------------------------

    def process_split(self, split):

        rows = []

        for label in ["autism", "control"]:

            subjects = (self.input_dir / split / label).iterdir()

            for subject in subjects:

                subject_id = subject.name

                aseg_file = subject / "stats" / "aseg.stats"

                if not aseg_file.exists():
                    continue

                features = self.parse_aseg(aseg_file)

                features["subject_id"] = subject_id
                features["label"] = label

                rows.append(features)

        df = pd.DataFrame(rows)

        df.to_csv(
            self.output_dir / f"{split}_morph_features.csv",
            index=False
        )

        print(split, "morph features shape:", df.shape)

    # --------------------------------
    # RUN
    # --------------------------------

    def run(self):

        self.process_split("train")
        self.process_split("test")

        print("Morphometric feature extraction completed")