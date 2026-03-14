import cv2
import joblib
import pandas as pd
import mlflow.sklearn
import nibabel as nib
import numpy as np
import os

from src.components.feature_extraction import FeatureExtraction
from dotenv import load_dotenv

load_dotenv()


class ASDInference:

    def __init__(self):

        print("Loading STAGING model from MLflow Registry...")

        mlflow.set_tracking_uri(
            "https://dagshub.com/renjini2539thomas/ASD_MLOps_Diagnoser.mlflow"
        )

        # ⭐ Load SKLEARN flavor model (VERY IMPORTANT)
        self.model = mlflow.sklearn.load_model(
            "models:/ASD_MRI_Diagnosis_Model@staging"
        )

        print("Model loaded successfully")

        print("Loading scaler...")
        self.scaler = joblib.load(
            "artifacts/scaled_features/scaler.pkl"
        )

        print("Loading selected features...")
        self.selected_features = joblib.load(
            "artifacts/feature_selection/selected_features.pkl"
        )

        self.extractor = FeatureExtraction()

    # -----------------------------------------------------

    def normalize(self, img):

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype("uint8")

        return img

    # -----------------------------------------------------

    def load_image(self, image_path):

        ext = os.path.splitext(image_path)[1].lower()

        # ⭐ MRI VOLUME (.nii / .mgz)
        if ext in [".nii", ".mgz", ".gz"]:

            nii = nib.load(image_path)
            vol = nii.get_fdata()

            mid_index = vol.shape[2] // 2
            slice_img = vol[:, :, mid_index]

            slice_img = self.normalize(slice_img)

            slice_img = cv2.resize(slice_img, (256, 256))

            return slice_img

        # ⭐ NORMAL IMAGE
        else:

            img = cv2.imread(image_path, 0)

            if img is None:
                raise ValueError("Invalid MRI image")

            img = cv2.resize(img, (256, 256))

            return img

    # -----------------------------------------------------

    def predict(self, image_path):

        img = self.load_image(image_path)

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

        # ⭐ PREDICTION
        pred = self.model.predict(X_selected)[0]

        probs = self.model.predict_proba(X_selected)[0]

        autism_index = list(self.model.classes_).index("autism")

        autism_probability = float(probs[autism_index])

        return pred, autism_probability