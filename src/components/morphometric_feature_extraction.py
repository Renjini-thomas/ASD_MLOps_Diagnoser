import pandas as pd
from pathlib import Path


class MorphometricFeatureExtraction:

    def __init__(self):

        self.input_dir = Path("data/processed")
        self.output_dir = Path("artifacts/morphometric_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------
    # PARSE ASEG (ROBUST)
    # --------------------------------

    def parse_aseg(self, file):

        features = {}

        with open(file) as f:
            for line in f:

                line = line.strip()

                # skip headers
                if not line or line.startswith("#"):
                    continue

                tokens = line.split()

                # skip non-table rows
                if not tokens[0].isdigit():
                    continue

                if len(tokens) < 5:
                    continue

                region = tokens[4]
                volume = float(tokens[3])

                if region == "Left-Hippocampus":
                    features["lh_hippo"] = volume

                elif region == "Right-Hippocampus":
                    features["rh_hippo"] = volume

                elif region == "Left-Amygdala":
                    features["lh_amyg"] = volume

                elif region == "Right-Amygdala":
                    features["rh_amyg"] = volume

                elif region == "Brain-Stem":
                    features["brain_stem"] = volume

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