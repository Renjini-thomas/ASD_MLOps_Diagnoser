import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


class FeatureSelection:

    def __init__(self):

        self.input_dir = Path("artifacts/features")
        self.output_dir = Path("artifacts/feature_selection")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # FIX: Use RFECV — automatically finds the optimal number of features
        # via cross-validation rather than a fixed arbitrary count
        self.min_features = 10

    def run(self):

        train_df = pd.read_csv(self.input_dir / "train_features.csv")
        test_df = pd.read_csv(self.input_dir / "test_features.csv")

        X_train = train_df.drop(["image_id", "label"], axis=1)
        y_train = train_df["label"]

        # Use a lighter RF for selection to reduce overfitting
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,           # FIX: limit depth to reduce overfitting
            class_weight="balanced",  # FIX: handle class imbalance
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": rf.feature_importances_
        })

        importance = importance.sort_values(
            by="importance",
            ascending=False
        )

        importance.to_csv(
            self.output_dir / "feature_importance.csv",
            index=False
        )

        # FIX: RFECV selects the optimal feature count automatically
        # using cross-validation — avoids arbitrary n_features choice
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        selector = RFECV(
            estimator=RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            step=5,
            cv=cv,
            scoring="f1_macro",    # Use F1 macro for class-balanced scoring
            min_features_to_select=self.min_features,
            n_jobs=-1
        )

        selector.fit(X_train, y_train)

        selected_features = X_train.columns[selector.support_]

        print(f"Optimal number of features selected: {len(selected_features)}")

        selected_df = pd.DataFrame({
            "feature": selected_features
        })

        selected_df.to_csv(
            self.output_dir / "selected_features.csv",
            index=False
        )

        train_selected = train_df[
            ["image_id"] + selected_features.tolist() + ["label"]
        ]

        test_selected = test_df[
            ["image_id"] + selected_features.tolist() + ["label"]
        ]

        train_selected.to_csv(
            self.output_dir / "train_selected.csv",
            index=False
        )

        test_selected.to_csv(
            self.output_dir / "test_selected.csv",
            index=False
        )