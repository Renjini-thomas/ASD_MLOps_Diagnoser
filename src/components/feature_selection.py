# import pandas as pd
# import numpy as np
# from pathlib import Path


# class FeatureSelection:

#     def __init__(self):

#         self.input_dir = Path("artifacts/aggregated_features")
#         self.output_dir = Path("artifacts/feature_selection")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         # correlation threshold (paper-style redundancy removal)
#         self.corr_threshold = 0.90


#     def remove_correlated_features(self, X):

#         corr_matrix = X.corr().abs()

#         upper = corr_matrix.where(
#             np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
#         )

#         to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]

#         print(f"Removing {len(to_drop)} highly correlated features")

#         return X.drop(columns=to_drop), to_drop


#     def run(self):

#         train_df = pd.read_csv(self.input_dir / "train_aggregated.csv")
#         test_df = pd.read_csv(self.input_dir / "test_aggregated.csv")

#         X_train = train_df.drop(["subject", "label"], axis=1)

#         # Cross-correlation filtering
#         X_filtered, dropped_features = self.remove_correlated_features(X_train)

#         selected_features = X_filtered.columns.tolist()

#         print(f"Remaining features after correlation filtering: {len(selected_features)}")

#         # save dropped + selected info
#         pd.DataFrame({"dropped_feature": dropped_features}).to_csv(
#             self.output_dir / "dropped_features.csv",
#             index=False
#         )

#         pd.DataFrame({"selected_feature": selected_features}).to_csv(
#             self.output_dir / "selected_features.csv",
#             index=False
#         )

#         # create final datasets
#         train_selected = train_df[
#             ["subject"] + selected_features + ["label"]
#         ]

#         test_selected = test_df[
#             ["subject"] + selected_features + ["label"]
#         ]

#         train_selected.to_csv(
#             self.output_dir / "train_selected.csv",
#             index=False
#         )

#         test_selected.to_csv(
#             self.output_dir / "test_selected.csv",
#             index=False
#         )
# import pandas as pd
# from pathlib import Path
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import StratifiedKFold


# class FeatureSelection:

#     def __init__(self):

#         self.input_dir = Path("artifacts/aggregated_features")
#         self.output_dir = Path("artifacts/feature_selection")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         # FIX: Use RFECV — automatically finds the optimal number of features
#         # via cross-validation rather than a fixed arbitrary count
#         self.min_features = 20

#     def run(self):

#         train_df = pd.read_csv(self.input_dir / "train_aggregated.csv")
#         test_df = pd.read_csv(self.input_dir / "test_aggregated.csv")

#         X_train = train_df.drop(["subject", "label"], axis=1)
#         y_train = train_df["label"]

#         # Use a lighter RF for selection to reduce overfitting
#         rf = RandomForestClassifier(
#             n_estimators=200,
#             max_depth=10,           # FIX: limit depth to reduce overfitting
#             class_weight="balanced",  # FIX: handle class imbalance
#             random_state=42,
#             n_jobs=-1
#         )

#         rf.fit(X_train, y_train)

#         importance = pd.DataFrame({
#             "feature": X_train.columns,
#             "importance": rf.feature_importances_
#         })

#         importance = importance.sort_values(
#             by="importance",
#             ascending=False
#         )

#         importance.to_csv(
#             self.output_dir / "feature_importance.csv",
#             index=False
#         )

#         # FIX: RFECV selects the optimal feature count automatically
#         # using cross-validation — avoids arbitrary n_features choice
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         selector = RFECV(
#             estimator=RandomForestClassifier(
#                 n_estimators=100,
#                 max_depth=8,
#                 class_weight="balanced",
#                 random_state=42,
#                 n_jobs=-1
#             ),
#             step=5,
#             cv=cv,
#             scoring="f1_macro",    # Use F1 macro for class-balanced scoring
#             min_features_to_select=self.min_features,
#             n_jobs=-1
#         )

#         selector.fit(X_train, y_train)

#         selected_features = X_train.columns[selector.support_]

#         print(f"Optimal number of features selected: {len(selected_features)}")

#         selected_df = pd.DataFrame({
#             "feature": selected_features
#         })

#         selected_df.to_csv(
#             self.output_dir / "selected_features.csv",
#             index=False
#         )

#         train_selected = train_df[
#             ["subject"] + selected_features.tolist() + ["label"]
#         ]

#         test_selected = test_df[
#             ["subject"] + selected_features.tolist() + ["label"]
#         ]

#         train_selected.to_csv(
#             self.output_dir / "train_selected.csv",
#             index=False
#         )

#         test_selected.to_csv(
#             self.output_dir / "test_selected.csv",
#             index=False
#         )
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy.stats import rankdata
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.feature_selection import RFECV, VarianceThreshold
# from sklearn.inspection import permutation_importance
# from sklearn.model_selection import StratifiedKFold


# class FeatureSelection:

#     def __init__(self):
#         self.input_dir = Path("artifacts/aggregated_features")
#         self.output_dir = Path("artifacts/feature_selection")
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.min_features = 20
#         self.correlation_threshold = 0.90
#         self.variance_threshold = 0.01

#     def remove_low_variance(self, X):
#         vt = VarianceThreshold(threshold=self.variance_threshold)
#         vt.fit(X)
#         kept = X.columns[vt.get_support()]
#         print(f"Variance filter: {X.shape[1]} → {len(kept)} features")
#         return X[kept]

#     def remove_correlated_features(self, X):
#         corr_matrix = X.corr().abs()
#         upper = corr_matrix.where(
#             np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
#         )
#         to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
#         print(f"Correlation filter: dropping {len(to_drop)} redundant features")
#         return X.drop(columns=to_drop)

#     def run(self):
#         train_df = pd.read_csv(self.input_dir / "train_aggregated.csv")
#         test_df = pd.read_csv(self.input_dir / "test_aggregated.csv")

#         X_train = train_df.drop(["subject", "label"], axis=1)
#         y_train = train_df["label"]

#         # Step 1: Remove near-zero variance features
#         X_train = self.remove_low_variance(X_train)

#         # Step 2: Remove highly correlated features
#         X_train = self.remove_correlated_features(X_train)

#         # Step 3: Fit RF + GB, aggregate importance rankings
#         rf = RandomForestClassifier(
#             n_estimators=200, max_depth=10,
#             class_weight="balanced", random_state=42, n_jobs=-1
#         )
#         rf.fit(X_train, y_train)

#         gb = GradientBoostingClassifier(
#             n_estimators=100, max_depth=4, random_state=42
#         )
#         gb.fit(X_train, y_train)

#         rf_ranks = rankdata(-rf.feature_importances_)
#         gb_ranks = rankdata(-gb.feature_importances_)
#         avg_rank = (rf_ranks + gb_ranks) / 2

#         # Step 4: Permutation importance (bias-corrected)
#         perm = permutation_importance(
#             rf, X_train, y_train,
#             n_repeats=10, scoring="f1_macro",
#             random_state=42, n_jobs=-1
#         )

#         importance = pd.DataFrame({
#             "feature": X_train.columns,
#             "rf_importance": rf.feature_importances_,
#             "gb_importance": gb.feature_importances_,
#             "avg_rank": avg_rank,
#             "perm_importance_mean": perm.importances_mean,
#             "perm_importance_std": perm.importances_std,
#         }).sort_values("avg_rank")

#         importance.to_csv(self.output_dir / "feature_importance.csv", index=False)

#         # Step 5: RFECV using GradientBoosting for cross-validated selection
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#         selector = RFECV(
#             estimator=GradientBoostingClassifier(
#                 n_estimators=100, max_depth=4, random_state=42
#             ),
#             step=5,
#             cv=cv,
#             scoring="f1_macro",
#             min_features_to_select=self.min_features,
#             n_jobs=-1
#         )
#         selector.fit(X_train, y_train)

#         selected_features = X_train.columns[selector.support_].tolist()
#         print(f"Optimal features selected: {len(selected_features)}")

#         pd.DataFrame({"feature": selected_features}).to_csv(
#             self.output_dir / "selected_features.csv", index=False
#         )

#         # Align test set to same filtered features
#         train_selected = train_df[["subject"] + selected_features + ["label"]]
#         test_selected = test_df[["subject"] + selected_features + ["label"]]

#         train_selected.to_csv(self.output_dir / "train_selected.csv", index=False)
#         test_selected.to_csv(self.output_dir / "test_selected.csv", index=False)
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from sklearn.feature_selection import mutual_info_classif


# class FeatureSelection:

#     def __init__(self):

#         self.input_dir = Path("artifacts/aggregated_features")
#         self.output_dir = Path("artifacts/feature_selection")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.corr_threshold = 0.90


#     # ------------------------------
#     # Remove Redundant Features
#     # ------------------------------
#     def remove_correlated_features(self, X):

#         corr_matrix = X.corr().abs()

#         upper = corr_matrix.where(
#             np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
#         )

#         to_drop = [
#             col for col in upper.columns
#             if any(upper[col] > self.corr_threshold)
#         ]

#         print(f"Removing {len(to_drop)} highly correlated features")

#         return X.drop(columns=to_drop), to_drop


#     # ------------------------------
#     # Mutual Information Ranking
#     # ------------------------------
#     def mutual_information_ranking(self, X, y):

#         y_numeric = y.map({"control": 0, "autism": 1})

#         mi_scores = mutual_info_classif(
#             X,
#             y_numeric,
#             random_state=42
#         )

#         ranking_df = pd.DataFrame({
#             "feature": X.columns,
#             "mi_score": mi_scores
#         })

#         ranking_df = ranking_df.sort_values(
#             by="mi_score",
#             ascending=False
#         )

#         print("\nTop 10 MI Features:")
#         print(ranking_df.head(10))

#         return ranking_df


#     # ------------------------------
#     # RUN
#     # ------------------------------
#     def run(self):

#         train_df = pd.read_csv(self.input_dir / "train_aggregated.csv")
#         test_df = pd.read_csv(self.input_dir / "test_aggregated.csv")

#         X_train = train_df.drop(["subject", "label"], axis=1)
#         y_train = train_df["label"]

#         # STEP 1 → Correlation Filtering
#         X_filtered, dropped = self.remove_correlated_features(X_train)

#         print(f"Remaining features after filtering: {X_filtered.shape[1]}")

#         # STEP 2 → Mutual Information Ranking
#         ranking_df = self.mutual_information_ranking(
#             X_filtered,
#             y_train
#         )

#         ranking_df.to_csv(
#             self.output_dir / "feature_ranking.csv",
#             index=False
#         )

#         selected_features = ranking_df["feature"].tolist()

#         pd.DataFrame({"feature": selected_features}).to_csv(
#             self.output_dir / "selected_features.csv",
#             index=False
#         )

#         pd.DataFrame({"dropped_feature": dropped}).to_csv(
#             self.output_dir / "dropped_features.csv",
#             index=False
#         )

#         # STEP 3 → Save filtered datasets
#         train_selected = train_df[
#             ["subject"] + selected_features + ["label"]
#         ]

#         test_selected = test_df[
#             ["subject"] + selected_features + ["label"]
#         ]

#         train_selected.to_csv(
#             self.output_dir / "train_selected.csv",
#             index=False
#         )

#         test_selected.to_csv(
#             self.output_dir / "test_selected.csv",
#             index=False
#         )
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib


class FeatureSelection:

    def __init__(self):

        self.input_dir = Path("artifacts/scaled_features")
        self.output_dir = Path("artifacts/feature_selection")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.top_k = 60

    def run(self):

        print("Loading aggregated features...")

        train_df = pd.read_csv(self.input_dir / "train_features.csv")
        test_df = pd.read_csv(self.input_dir / "test_features.csv")

        X_train = train_df.drop(["subject_id", "label"], axis=1)
        y_train = train_df["label"]

        X_test = test_df.drop(["subject_id", "label"], axis=1)
        y_test = test_df["label"]

        print("Training RandomForest for feature ranking...")

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        importance = pd.Series(
            rf.feature_importances_,
            index=X_train.columns
        )

        ranked_features = importance.sort_values(ascending=False)

        print("Top features:\n", ranked_features.head())

        selected_features = ranked_features.head(self.top_k).index
        joblib.dump(selected_features, self.output_dir / "selected_features.pkl")

        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]

        train_selected = pd.concat(
            [train_df[["subject_id", "label"]], X_train_sel],
            axis=1
        )

        test_selected = pd.concat(
            [test_df[["subject_id", "label"]], X_test_sel],
            axis=1
        )

        train_selected.to_csv(
            self.output_dir / "train_selected.csv",
            index=False
        )

        test_selected.to_csv(
            self.output_dir / "test_selected.csv",
            index=False
        )

        ranked_features.to_csv(
            self.output_dir / "feature_ranking.csv"
        )

#         print("Feature selection completed")
# import pandas as pd
# from pathlib import Path

# from sklearn.feature_selection import RFECV
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold


# class FeatureSelection:

#     def __init__(self):

#         self.input_dir = Path("artifacts/scaled_features")
#         self.output_dir = Path("artifacts/feature_selection")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     # ------------------------------------------------

#     def run(self):

#         print("Loading scaled hybrid features...")

#         train_df = pd.read_csv(self.input_dir / "train_features.csv")
#         test_df = pd.read_csv(self.input_dir / "test_features.csv")

#         X_train = train_df.drop(["subject_id", "label"], axis=1)
#         y_train = train_df["label"]

#         X_test = test_df.drop(["subject_id", "label"], axis=1)
#         y_test = test_df["label"]

#         print("Running RFECV feature selection...")

#         estimator = LogisticRegression(
#             max_iter=2000,
#             class_weight="balanced",
#             solver="liblinear"
#         )

#         cv = StratifiedKFold(
#             n_splits=5,
#             shuffle=True,
#             random_state=42
#         )

#         selector = RFECV(
#             estimator=estimator,
#             step=1,
#             cv=cv,
#             scoring="f1_macro",
#             n_jobs=-1,
#             min_features_to_select=10
#         )

#         selector.fit(X_train, y_train)

#         selected_features = X_train.columns[selector.support_]

#         print("Optimal number of features:", selector.n_features_)
#         print("Selected features:", list(selected_features))

#         # ---------------- SAVE RANKING ----------------

#         ranking_df = pd.DataFrame({
#             "feature": X_train.columns,
#             "rank": selector.ranking_
#         }).sort_values("rank")

#         ranking_df.to_csv(
#             self.output_dir / "feature_ranking.csv",
#             index=False
#         )

#         # ---------------- SAVE SELECTED DATASET ----------------

#         X_train_sel = X_train[selected_features]
#         X_test_sel = X_test[selected_features]

#         train_selected = pd.concat(
#             [train_df[["subject_id", "label"]], X_train_sel],
#             axis=1
#         )

#         test_selected = pd.concat(
#             [test_df[["subject_id", "label"]], X_test_sel],
#             axis=1
#         )

#         train_selected.to_csv(
#             self.output_dir / "train_selected.csv",
#             index=False
#         )

#         test_selected.to_csv(
#             self.output_dir / "test_selected.csv",
#             index=False
#         )

#         print("RFECV Feature Selection completed successfully")
# import pandas as pd
# from pathlib import Path

# from sklearn.feature_selection import RFECV
# from sklearn.svm import SVC


# class FeatureSelection:

#     def __init__(self):

#         self.input_dir = Path("artifacts/scaled_features")
#         self.output_dir = Path("artifacts/feature_selection")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     def run(self):

#         print("Loading scaled features...")

#         train_df = pd.read_csv(self.input_dir / "train_features.csv")
#         test_df = pd.read_csv(self.input_dir / "test_features.csv")

#         X_train = train_df.drop(["subject_id", "label"], axis=1)
#         y_train = train_df["label"]

#         X_test = test_df.drop(["subject_id", "label"], axis=1)
#         y_test = test_df["label"]

#         print("Running RFECV feature selection...")

#         estimator = SVC(kernel="linear")

#         selector = RFECV(
#             estimator=estimator,
#             step=1,
#             cv=5,
#             scoring="accuracy",
#             n_jobs=-1
#         )

#         selector.fit(X_train, y_train)

#         selected_features = X_train.columns[selector.support_]

#         print("Optimal number of features:", selector.n_features_)
#         print("Selected features:", list(selected_features))

#         # ⭐ Transform datasets
#         X_train_sel = X_train[selected_features]
#         X_test_sel = X_test[selected_features]

#         # ⭐ Save selected datasets
#         train_selected = pd.concat(
#             [train_df[["subject_id", "label"]], X_train_sel],
#             axis=1
#         )

#         test_selected = pd.concat(
#             [test_df[["subject_id", "label"]], X_test_sel],
#             axis=1
#         )

#         train_selected.to_csv(
#             self.output_dir / "train_selected.csv",
#             index=False
#         )

#         test_selected.to_csv(
#             self.output_dir / "test_selected.csv",
#             index=False
#         )

#         # ⭐ Save ranking info
#         ranking_df = pd.DataFrame({
#             "feature": X_train.columns,
#             "ranking": selector.ranking_,
#             "selected": selector.support_
#         })

#         ranking_df.to_csv(
#             self.output_dir / "feature_ranking_rfecv.csv",
#             index=False
#         )

#         print("Feature selection completed successfully")