import cv2
import joblib
import pandas as pd
import mlflow.pyfunc

from src.components.feature_extraction import FeatureExtraction
from dotenv import load_dotenv
load_dotenv()

class ASDInference:

    def __init__(self):

        print("Loading staging model...")
        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/ASD_MLOps_Diagnoser.mlflow"
        )
        self.model = mlflow.pyfunc.load_model(
            "models:/ASD_MRI_Diagnosis_Model@staging"
        )

        print("Loading scaler...")
        self.scaler = joblib.load(
            "artifacts/scaled_features/scaler.pkl"
        )

        print("Loading selected features...")
        self.selected_features = joblib.load(
            "artifacts/feature_selection/selected_features.pkl"
        )

        self.extractor = FeatureExtraction()

    # ------------------------------------

    def predict(self, image_path):

        img = cv2.imread(image_path, 0)

        if img is None:
            raise ValueError("Invalid MRI")

        # FEATURE EXTRACTION
        features = (
            self.extractor.glcm_features(img)
            + self.extractor.lbp_features(img)
            + self.extractor.histogram_features(img)
            + self.extractor.wavelet_features(img)
        )

        X = pd.DataFrame(
            [features],
            columns=self.extractor.get_feature_names()
        )

        # SCALING
        X_scaled = self.scaler.transform(X)

        X_scaled = pd.DataFrame(
            X_scaled,
            columns=self.extractor.get_feature_names()
        )

        # FEATURE SELECTION
        X_selected = X_scaled[self.selected_features]

        # PREDICTION
        pred = self.model.predict(X_selected)[0]

        prob = self.model.predict_proba(X_selected).max()

        return pred, prob