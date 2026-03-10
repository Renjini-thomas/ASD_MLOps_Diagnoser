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

        self.dataset_name = "ABIDE1_MidSagittal_Features"

        self.remote_uri = "https://dagshub.com/renjini2539thomas/ASD_MLOps_Diagnoser.mlflow"
        self.local_uri = "file:./mlruns"

    # ------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------

    def load_data(self):

        train_df = pd.read_csv(self.input_dir / "train_selected.csv")
        test_df = pd.read_csv(self.input_dir / "test_selected.csv")

        X_train = train_df.drop(["image_id", "label"], axis=1)
        y_train = train_df["label"]

        X_test = test_df.drop(["image_id", "label"], axis=1)
        y_test = test_df["label"]

        print(f"Train class distribution:\n{y_train.value_counts()}")
        print(f"Test class distribution:\n{y_test.value_counts()}")

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------
    # MODEL EVALUATION
    # ------------------------------------------------

    def evaluate(self, model, X_train, y_train, X_test, y_test):

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        precision = precision_score(y_test, test_preds, pos_label="autism", zero_division=0)
        recall = recall_score(y_test, test_preds, pos_label="autism", zero_division=0)
        f1 = f1_score(y_test, test_preds, pos_label="autism", zero_division=0)

        probs = model.predict_proba(X_test)[:, 1]
        y_test_binary = (y_test == "autism").astype(int)
        auc = roc_auc_score(y_test_binary, probs)

        return train_acc, test_acc, precision, recall, f1, auc

    # ------------------------------------------------
    # COMPOSITE SCORING FOR BEST MODEL SELECTION
    # ------------------------------------------------

    def composite_score(self, f1, recall, precision, auc):
        return 0.4 * f1 + 0.3 * recall + 0.2 * precision + 0.1 * auc

    # ------------------------------------------------
    # TRAINING PIPELINE
    # ------------------------------------------------

    def run(self):

        X_train, X_test, y_train, y_test = self.load_data()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # -----------------------------------------------
        # Model definitions with hyperparameter grids
        # -----------------------------------------------

        model_configs = {

            # SVM variants — class_weight='balanced' handles imbalance
            "SVM_rbf": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", probability=True, class_weight="balanced"))
                ]),
                "param_grid": {
                    "model__C": [0.1, 1, 10, 100],
                    "model__gamma": ["scale", "auto", 0.001, 0.01]
                }
            },

            "SVM_linear": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="linear", probability=True, class_weight="balanced"))
                ]),
                "param_grid": {
                    "model__C": [0.01, 0.1, 1, 10]
                }
            },

            # KNN — best k via CV
            "KNN": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier())
                ]),
                "param_grid": {
                    "model__n_neighbors": [3, 5, 7, 9, 11],
                    "model__weights": ["uniform", "distance"],
                    "model__metric": ["euclidean", "manhattan"]
                }
            },

            # Random Forest — strong ensemble, handles imbalance natively
            "RandomForest": {
                "pipeline": Pipeline([
                    ("model", RandomForestClassifier(
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1
                    ))
                ]),
                "param_grid": {
                    "model__n_estimators": [200, 500],
                    "model__max_depth": [5, 10, None],
                    "model__min_samples_split": [2, 5]
                }
            },

            # Gradient Boosting — often best on tabular medical data
            "GradientBoosting": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingClassifier(random_state=42))
                ]),
                "param_grid": {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1, 0.2],
                    "model__max_depth": [3, 5]
                }
            },

            # Logistic Regression — strong baseline for balanced features
            "LogisticRegression": {
                "pipeline": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42
                    ))
                ]),
                "param_grid": {
                    "model__C": [0.01, 0.1, 1, 10],
                    "model__penalty": ["l1", "l2"],
                    "model__solver": ["liblinear"]
                }
            },

            # Decision Tree with depth limit to combat overfitting
            "DecisionTree": {
                "pipeline": Pipeline([
                    ("model", DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=42
                    ))
                ]),
                "param_grid": {
                    "model__max_depth": [3, 5, 7, 10],
                    "model__min_samples_split": [2, 5, 10],
                    "model__criterion": ["gini", "entropy"]
                }
            },
        }

        results = []
        best_score = -1
        best_model = None
        best_model_name = None

        for name, config in model_configs.items():

            print(f"\nTuning {name}...")

            # Grid search with stratified CV and F1 macro scoring
            grid_search = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["param_grid"],
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                refit=True,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            print(f"  Best params: {grid_search.best_params_}")

            train_acc, test_acc, precision, recall, f1, auc = self.evaluate(
                model, X_train, y_train, X_test, y_test
            )

            print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            # ----------------------------
            # LOCAL MLFLOW LOGGING
            # ----------------------------

            mlflow.set_tracking_uri(self.local_uri)
            mlflow.set_experiment("ASD_Local_Experiments")

            with mlflow.start_run(run_name=name):

                mlflow.log_param("dataset", self.dataset_name)
                mlflow.log_param("model", name)
                mlflow.log_params({
                    f"best_{k}": v for k, v in grid_search.best_params_.items()
                })

                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("auc", auc)

                mlflow.sklearn.log_model(model, name)

            # ----------------------------
            # DAGSHUB MLFLOW LOGGING
            # ----------------------------

            mlflow.set_tracking_uri(self.remote_uri)
            mlflow.set_experiment("ASD_Diagnosis_Experiments")

            with mlflow.start_run(run_name=name):

                mlflow.log_param("dataset", self.dataset_name)
                mlflow.log_param("model", name)
                mlflow.log_params({
                    f"best_{k}": v for k, v in grid_search.best_params_.items()
                })

                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("auc", auc)

                mlflow.sklearn.log_model(model, name)

            results.append({
                "model": name,
                "best_params": str(grid_search.best_params_),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc": auc
            })

            # FIX: Also track best_accuracy correctly
            score = self.composite_score(f1, recall, precision, auc)
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
                best_test_accuracy = test_acc  # FIX: was never set before

        # Save results
        results_df = pd.DataFrame(results)
        results_path = self.output_dir / "model_results.csv"
        results_df.to_csv(results_path, index=False)

        # Save best model
        best_model_path = self.output_dir / "best_model.pkl"
        joblib.dump(best_model, best_model_path)

        print("\n" + "=" * 50)
        print(f"Best Model      : {best_model_name}")
        print(f"Best Test Acc   : {best_test_accuracy:.4f}")
        print(f"Composite Score : {best_score:.4f}")
        print("=" * 50)

        return results_df