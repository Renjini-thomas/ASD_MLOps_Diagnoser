import pandas as pd
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from dotenv import load_dotenv

load_dotenv()


class ModelTraining:

    def __init__(self):

        self.input_dir = Path("artifacts/feature_selection")
        self.output_dir = Path("artifacts/model_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_name = "ABIDE_Hybrid_MidSagittal"

        self.local_uri = "file:./mlruns"
        self.remote_uri = "https://dagshub.com/renjini2539thomas/ASD_MLOps_Diagnoser.mlflow"

        self.feature_selection_method = "RandomForest Importance"

    # ------------------------------------------------

    def load_data(self):

        train_df = pd.read_csv(self.input_dir / "train_features.csv")
        test_df = pd.read_csv(self.input_dir / "test_features.csv")

        X_train = train_df.drop(["subject_id", "label"], axis=1)
        y_train = train_df["label"]

        X_test = test_df.drop(["subject_id", "label"], axis=1)
        y_test = test_df["label"]

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------

    def evaluate(self, model, X_test, y_test):

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, pos_label="autism", zero_division=0)
        recall = recall_score(y_test, preds, pos_label="autism", zero_division=0)
        f1 = f1_score(y_test, preds, pos_label="autism", zero_division=0)

        probs = model.predict_proba(X_test)[:, 1]
        y_bin = (y_test == "autism").astype(int)

        auc = roc_auc_score(y_bin, probs)

        return acc, precision, recall, f1, auc

    # ------------------------------------------------

    def composite_score(self, f1, auc):
        return 0.6 * f1 + 0.4 * auc

    # ------------------------------------------------

    def run(self):

        X_train, X_test, y_train, y_test = self.load_data()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        model_configs = {

            "SVM_rbf": {
                "pipeline": Pipeline([
                    
                    ("model", SVC(kernel="rbf", probability=True, class_weight="balanced"))
                ]),
                "params": {
                    "model__C": [0.1, 1, 10],
                    "model__gamma": ["scale", 0.01]
                }
            },

            "SVM_linear": {
                "pipeline": Pipeline([
                    
                    ("model", SVC(kernel="linear", probability=True, class_weight="balanced"))
                ]),
                "params": {
                    "model__C": [0.01, 0.1, 1]
                }
            },

            "KNN": {
                "pipeline": Pipeline([
                    
                    ("model", KNeighborsClassifier())
                ]),
                "params": {
                    "model__n_neighbors": [3, 5, 7],
                    "model__weights": ["uniform", "distance"]
                }
            },

            "GradientBoosting": {
                "pipeline": Pipeline([
                    ("model", GradientBoostingClassifier(random_state=42))
                ]),
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1]
                }
            },

            "LogisticRegression": {
                "pipeline": Pipeline([
                    
                    ("model", LogisticRegression(class_weight="balanced", max_iter=2000))
                ]),
                "params": {
                    "model__C": [0.1, 1, 10]
                }
            },

            "DecisionTree": {
                "pipeline": Pipeline([
                    ("model", DecisionTreeClassifier(class_weight="balanced", random_state=42))
                ]),
                "params": {
                    "model__max_depth": [3, 5, 7]
                }
            }
        }

        best_score = -1
        best_model = None
        best_name = None

        results = []

        for name, cfg in model_configs.items():

            grid = GridSearchCV(
                cfg["pipeline"],
                cfg["params"],
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            model = grid.best_estimator_

            acc, precision, recall, f1, auc = self.evaluate(model, X_test, y_test)

            score = self.composite_score(f1, auc)

            results.append({
                "dataset": self.dataset_name,
                "model": name,
                "best_params": str(grid.best_params_),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "composite_score": score
            })

            # ---------- LOCAL LOG ----------
            mlflow.set_tracking_uri(self.local_uri)
            mlflow.set_experiment("ASD_Local")

            with mlflow.start_run(run_name=name):

                mlflow.log_param("dataset", self.dataset_name)
                mlflow.log_param("model", name)
                mlflow.log_param("feature_selection", self.feature_selection_method)
                mlflow.log_params(grid.best_params_)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("auc", auc)

                mlflow.sklearn.log_model(model, name)

            # ---------- DAGSHUB LOG ----------
            mlflow.set_tracking_uri(self.remote_uri)
            mlflow.set_experiment("ASD_Remote")

            with mlflow.start_run(run_name=name):

                mlflow.log_param("dataset", self.dataset_name)
                mlflow.log_param("model", name)
                mlflow.log_param("feature_selection", self.feature_selection_method)
                mlflow.log_params(grid.best_params_)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("auc", auc)

                mlflow.sklearn.log_model(model, name)

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "model_results.csv", index=False)

        joblib.dump({
            "model": best_model,
            "features": X_train.columns.tolist(),
            "model_name": best_name,
            "dataset": self.dataset_name
        },
            self.output_dir / "best_model.pkl"
        )

        print("\nBEST MODEL:", best_name)
        print("Composite Score:", best_score)