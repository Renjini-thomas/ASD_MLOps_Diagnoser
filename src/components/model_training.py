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
from sklearn.metrics import make_scorer, f1_score,accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score  

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

        train_df = pd.read_csv(self.input_dir / "train_selected.csv")

        X_train = train_df.drop(["subject_id", "label"], axis=1)
        y_train = train_df["label"]

        return X_train, y_train

    # ------------------------------------------------

    def run(self):

        X_train, y_train = self.load_data()

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

            "RandomForest": {
                "pipeline": Pipeline([
                    ("model", RandomForestClassifier(
                        class_weight="balanced",
                        random_state=42
                    ))
                ]),
                "params": {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10]
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
                    
                    ("model", LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000
                    ))
                ]),
                "params": {
                    "model__C": [0.1, 1, 10]
                }
            },

            "DecisionTree": {
                "pipeline": Pipeline([
                    ("model", DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=42
                    ))
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
        f1_autism = make_scorer(f1_score, pos_label="autism")
        for name, cfg in model_configs.items():

            grid = GridSearchCV(
                cfg["pipeline"],
                cfg["params"],
                cv=cv,
                scoring=f1_autism,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            cv_score = grid.best_score_
            model = grid.best_estimator_

            results.append({
                "dataset": self.dataset_name,
                "model": name,
                "best_params": str(grid.best_params_),
                "cv_f1_score": cv_score
            })

            # ---------- LOCAL LOG ----------
            mlflow.set_tracking_uri(self.local_uri)
            mlflow.set_experiment("ASD_Training_Local")

            with mlflow.start_run(run_name=name):

                mlflow.log_param("dataset", self.dataset_name)
                mlflow.log_param("model", name)
                mlflow.log_param("feature_selection", self.feature_selection_method)
                mlflow.log_params(grid.best_params_)

                mlflow.log_metric("cv_f1", cv_score)

                mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
            )
                

            # ---------- DAGSHUB LOG ----------
            mlflow.set_tracking_uri(self.remote_uri)
            mlflow.set_experiment("ASD_Training_Remote")

            with mlflow.start_run(run_name=name):

                mlflow.log_param("dataset", self.dataset_name)
                mlflow.log_param("model", name)
                mlflow.log_param("feature_selection", self.feature_selection_method)
                mlflow.log_params(grid.best_params_)

                mlflow.log_metric("cv_f1", cv_score)

                mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
                )

            if cv_score > best_score:
                best_score = cv_score
                best_model = model
                best_name = name

        pd.DataFrame(results).to_csv(
            self.output_dir / "cv_model_results.csv",
            index=False
        )

        joblib.dump({
            "model": best_model,
            "features": X_train.columns.tolist(),
            "model_name": best_name,
            "dataset": self.dataset_name
        },
            self.output_dir / "best_model.pkl"
        )

        print("\nBEST MODEL:", best_name)
        print("Best CV F1:", best_score)