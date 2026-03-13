import pandas as pd
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv
load_dotenv()


class ModelEvaluation:

    def __init__(self):

        self.model_path = Path("artifacts/model_training/best_model.pkl")
        self.data_path = Path("artifacts/feature_selection/test_selected.csv")

        self.output_dir = Path("artifacts/model_evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.local_uri = "file:./mlruns"
        self.remote_uri = "https://dagshub.com/renjini2539thomas/ASD_MLOps_Diagnoser.mlflow"

        self.model_registry_name = "ASD_MRI_Diagnosis_Model"

        # MEDICAL SAFETY THRESHOLD
        self.min_recall_required = 0.60

    # ------------------------------------------------

    def load_data(self):

        df = pd.read_csv(self.data_path)

        X = df.drop(["subject_id", "label"], axis=1)
        y = df["label"]

        return X, y

    # ------------------------------------------------

    def compute_metrics(self, model, X_test, y_test):

        preds = model.predict(X_test)

        autism_index = list(model.classes_).index("autism")
        probs = model.predict_proba(X_test)[:, autism_index]

        y_binary = (y_test == "autism").astype(int)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label="autism", zero_division=0)
        recall = recall_score(y_test, preds, pos_label="autism", zero_division=0)
        f1 = f1_score(y_test, preds, pos_label="autism", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, preds)
        auc = roc_auc_score(y_binary, probs)

        return preds, probs, accuracy, precision, recall, f1, balanced_acc, auc

    # ------------------------------------------------

    def promote_if_better(self, client, new_run_id, recall, f1):

        try:
            staging = client.get_model_version_by_alias(
                self.model_registry_name,
                "staging"
            )

            prev_run = client.get_run(staging.run_id)

            prev_recall = prev_run.data.metrics.get("recall", 0)
            prev_f1 = prev_run.data.metrics.get("f1", 0)

            print("\nPrevious staging Recall:", prev_recall)
            print("Previous staging F1:", prev_f1)

            if recall < self.min_recall_required:
                print("\nNew model rejected ❌ (recall below clinical threshold)")
                return

            if (recall > prev_recall) or (
                recall == prev_recall and f1 > prev_f1
            ):
                latest = client.search_model_versions(
                    f"name='{self.model_registry_name}'"
                )[0]

                client.set_registered_model_alias(
                    name=self.model_registry_name,
                    alias="staging",
                    version=latest.version
                )

                print(f"\nModel promoted to STAGING ✅ (version {latest.version})")

            else:
                print("\nModel NOT promoted (not better than current staging)")

        except Exception:

            latest = client.search_model_versions(
                f"name='{self.model_registry_name}'"
            )[0]

            client.set_registered_model_alias(
                name=self.model_registry_name,
                alias="staging",
                version=latest.version
            )

            print("\nNo previous staging model → First model set as STAGING ✅")

    # ------------------------------------------------

    def run(self):

        model_package = joblib.load(self.model_path)

        model = model_package["model"]
        feature_names = model_package["features"]
        model_name = model_package["model_name"]
        dataset_name = model_package["dataset"]

        X_test, y_test = self.load_data()
        X_test = X_test[feature_names]

        preds, probs, accuracy, precision, recall, f1, balanced_acc, auc = \
            self.compute_metrics(model, X_test, y_test)

        labels = model.classes_
        cm = confusion_matrix(y_test, preds, labels=labels)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=labels,
                    yticklabels=labels,
                    cmap="Blues")

        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        y_binary = (y_test == "autism").astype(int)
        fpr, tpr, _ = roc_curve(y_binary, probs)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.legend()

        roc_path = self.output_dir / "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

        report_path = self.output_dir / "classification_report.txt"
        with open(report_path,"w") as f:
            f.write(classification_report(y_test, preds))

        mlflow.set_tracking_uri(self.remote_uri)
        mlflow.set_experiment("ASD_Model_Evaluation")

        with mlflow.start_run(run_name=model_name) as run:

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("balanced_accuracy", balanced_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("auc", auc)

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(report_path))

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.model_registry_name
            )

            run_id = run.info.run_id

        client = MlflowClient(tracking_uri=self.remote_uri)

        self.promote_if_better(client, run_id, recall, f1)

        print("\nEvaluation Completed")
        print("Recall:", recall)
        print("F1:", f1)
        print("AUC:", auc)