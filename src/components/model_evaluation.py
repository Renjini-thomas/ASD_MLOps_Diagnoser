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

        self.remote_uri = "https://dagshub.com/renjini2539thomas/ASD_MLOps_Diagnoser.mlflow"

        self.model_name = "ASD_MRI_Diagnosis_Model"

    # -------------------------
    # LOAD DATA
    # -------------------------

    def load_data(self):

        df = pd.read_csv(self.data_path)

        X = df.drop(["subject", "label"], axis=1)
        y = df["label"]

        return X, y

    # -------------------------
    # EVALUATION
    # -------------------------

    def run(self):

        model_data = joblib.load(self.model_path)

        model = model_data["model"]
        feature_names = model_data["features"]

        X_test, y_test = self.load_data()

        # Ensure correct feature order
        X_test = X_test[feature_names]

        preds = model.predict(X_test)

        classes = model.classes_
        autism_index = list(classes).index("autism")

        probs = model.predict_proba(X_test)[:, autism_index]

        y_binary = (y_test == "autism").astype(int)

        # -------------------------
        # METRICS
        # -------------------------

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label="autism", zero_division=0)
        recall = recall_score(y_test, preds, pos_label="autism", zero_division=0)
        f1 = f1_score(y_test, preds, pos_label="autism", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, preds)
        auc = roc_auc_score(y_binary, probs)

        # -------------------------
        # CONFUSION MATRIX
        # -------------------------

        cm = confusion_matrix(y_test, preds)

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Control", "Autism"],
            yticklabels=["Control", "Autism"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        # -------------------------
        # ROC CURVE
        # -------------------------

        fpr, tpr, _ = roc_curve(y_binary, probs)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        roc_path = self.output_dir / "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

        # -------------------------
        # CLASSIFICATION REPORT
        # -------------------------

        report = classification_report(y_test, preds, zero_division=0)

        report_path = self.output_dir / "classification_report.txt"

        with open(report_path, "w") as f:
            f.write(report)

        # -------------------------
        # MLFLOW LOGGING
        # -------------------------

        mlflow.set_tracking_uri(self.remote_uri)
        mlflow.set_experiment("ASD_Model_Evaluation")

        with mlflow.start_run(run_name="evaluation"):

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("balanced_accuracy", balanced_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc", auc)

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(report_path))

            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=self.model_name
            )

        # -------------------------
        # MODEL REGISTRY
        # -------------------------

        client = MlflowClient(tracking_uri=self.remote_uri)

        latest_versions = client.get_latest_versions(self.model_name)

        if latest_versions:
            latest_version = latest_versions[0].version

            client.set_registered_model_alias(
                name=self.model_name,
                alias="staging",
                version=latest_version
            )

            print(f"Model v{latest_version} aliased as 'staging'")

        print("\nModel Evaluation Completed")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"AUC       : {auc:.4f}")
        print("Model registered in MLflow Model Registry.")

