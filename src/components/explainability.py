import shap
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

from pathlib import Path
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()


class ModelExplainability:

    def __init__(self):

        self.tracking_uri = \
            "https://dagshub.com/renjini2539thomas/ASD_MLOps_Diagnoser.mlflow"

        self.registry_name = "ASD_MRI_Diagnosis_Model"

        self.meta_path = Path("artifacts/explainability/meta.json")

        self.data_path = Path(
            "artifacts/feature_selection/train_selected.csv"
        )

        self.output_dir = Path("artifacts/explainability")

    # -----------------------------------------------------

    def get_staging_version(self):

        client = MlflowClient(tracking_uri=self.tracking_uri)

        mv = client.get_model_version_by_alias(
            self.registry_name,
            "staging"
        )

        return mv.version

    # -----------------------------------------------------

    def should_run(self, version):

        if not self.meta_path.exists():
            return True

        with open(self.meta_path) as f:
            meta = json.load(f)

        return meta.get("last_explained_version") != version

    # -----------------------------------------------------

    def save_meta(self, version):

        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.meta_path, "w") as f:
            json.dump(
                {"last_explained_version": version},
                f
            )

    # -----------------------------------------------------

    def run(self):

        # ⭐ ensure DVC output path always exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        version = self.get_staging_version()

        print("Current STAGING version:", version)

        if not self.should_run(version):

            print("Explainability already computed → Skipping")

            marker = self.output_dir / "skip_marker.txt"
            marker.write_text("Explainability skipped")

            return

        print("Running Explainability...")

        mlflow.set_tracking_uri(self.tracking_uri)

        model = mlflow.sklearn.load_model(
            f"models:/{self.registry_name}/{version}"
        )

        df = pd.read_csv(self.data_path)

        X = df.drop(["subject_id", "label"], axis=1)

        # ⭐ FAST KernelSHAP background
        background = shap.sample(X, 30, random_state=42)

        # ⭐ wrapper to keep feature names
        def model_fn(x):
            x_df = pd.DataFrame(x, columns=background.columns)
            return model.predict_proba(x_df)

        explainer = shap.KernelExplainer(
            model_fn,
            background.values
        )

        shap_values = explainer.shap_values(
            background.values,
            nsamples=50
        )

        # ⭐ -------- ROBUST SHAP FORMAT HANDLING --------

        if isinstance(shap_values, list):
            sv = shap_values[1]       # autism class
        else:
            sv = shap_values

        sv = np.array(sv)

        # if returned tensor → collapse class axis
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        # strict feature alignment
        sv = sv[:, :background.shape[1]]

        # ⭐ ---------------------------------------------

        plt.figure()

        shap.summary_plot(
            sv,
            background,
            show=False
        )

        save_path = self.output_dir / \
            f"global_shap_summary_v{version}.png"

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        # ⭐ log explainability experiment
        mlflow.set_experiment("ASD_Model_Explainability")

        with mlflow.start_run(run_name=f"explain_v{version}"):

            mlflow.log_param("model_version", version)
            mlflow.log_artifact(str(save_path))

        self.save_meta(version)

        print("Explainability Completed for Version:", version)