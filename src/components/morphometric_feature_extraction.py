import pandas as pd
from pathlib import Path


class MorphometricFeatureExtraction:

    def __init__(self):

        self.input_dir = Path("data/processed")
        self.output_dir = Path("artifacts/morphometric_features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------
    # PARSE ASEG
    # --------------------------------

    def parse_aseg(self, file):

        features = {}

        with open(file) as f:
            for line in f:

                if "Left-Hippocampus" in line:
                    features["lh_hippo"] = float(line.split()[3])

                if "Right-Hippocampus" in line:
                    features["rh_hippo"] = float(line.split()[3])

                if "Left-Amygdala" in line:
                    features["lh_amyg"] = float(line.split()[3])

                if "Right-Amygdala" in line:
                    features["rh_amyg"] = float(line.split()[3])

                if "BrainSegVol" in line:
                    features["brain_vol"] = float(line.split()[1])

        return features

    # --------------------------------
    # PARSE APARC
    # --------------------------------

    def parse_aparc(self, file):

        thickness = None
        area = None

        with open(file) as f:
            for line in f:

                if "MeanThickness" in line:
                    thickness = float(line.split()[1])

                if "SurfArea" in line:
                    area = float(line.split()[1])

        return thickness, area

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
                lh_file = subject / "stats" / "lh.aparc.stats"
                rh_file = subject / "stats" / "rh.aparc.stats"

                if not aseg_file.exists():
                    continue

                features = self.parse_aseg(aseg_file)

                lh_thick, lh_area = self.parse_aparc(lh_file)
                rh_thick, rh_area = self.parse_aparc(rh_file)

                features["lh_thickness"] = lh_thick
                features["rh_thickness"] = rh_thick
                features["lh_area"] = lh_area
                features["rh_area"] = rh_area

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