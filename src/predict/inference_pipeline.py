import numpy as np
import cv2
import nibabel as nib
import pandas as pd
import joblib
from pathlib import Path

from src.components.feature_extraction import FeatureExtraction


class InferencePipeline:

    def __init__(self):

        self.model = joblib.load("artifacts/model_training/best_model.pkl")

        selected = pd.read_csv("artifacts/feature_selection/selected_features.csv")
        self.selected_features = selected["feature"].tolist()

        self.extractor = FeatureExtraction()

    # ----------------------------
    # LOAD IMAGE
    # ----------------------------

    def load_image(self, file_path):

        ext = Path(file_path).suffix.lower()

        if ext in [".mgz", ".nii", ".nii.gz"]:

            img = nib.load(file_path)
            data = img.get_fdata()

            mid = data.shape[0] // 2
            img = data[mid, :, :]

        else:

            img = cv2.imread(file_path, 0)

        return img

    # ----------------------------
    # PREPROCESS
    # ----------------------------

    def preprocess(self, img):

        img = cv2.resize(img, (256, 256))

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = (img * 255).astype(np.uint8)

        return img

    # ----------------------------
    # FEATURE PIPELINE
    # ----------------------------

    def extract_features(self, img):

        glcm = self.extractor.glcm_features(img)
        lbp = self.extractor.lbp_features(img)
        gfcc = self.extractor.gfcc_features(img)

        features = glcm + lbp + gfcc

        feature_names = self.extractor.get_feature_names()

        df = pd.DataFrame([features], columns=feature_names)

        return df

    # ----------------------------
    # PREDICT
    # ----------------------------

    def predict(self, file_path):

        img = self.load_image(file_path)

        img = self.preprocess(img)

        features = self.extract_features(img)

        features = features[self.selected_features]

        prediction = self.model.predict(features)[0]

        prob = self.model.predict_proba(features)[0]

        return prediction, prob