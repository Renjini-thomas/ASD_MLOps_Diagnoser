import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops


class FeatureExtraction:

    def __init__(self):

        self.input_dir = Path("data/preprocessed")
        self.output_dir = Path("artifacts/features")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # FEATURE NAMES
    # ------------------------------------------------

    def get_feature_names(self):

        glcm_names = [
            "glcm_contrast",
            "glcm_correlation",
            "glcm_energy",
            "glcm_homogeneity",
            "glcm_asm",
            "glcm_dissimilarity",
            "glcm_mean",
            "glcm_variance",
            "glcm_entropy",
            "glcm_cluster_shade",
            "glcm_cluster_prominence",
            "glcm_max_probability"
        ]

        # uniform LBP has 10 uniform patterns for P=8, R=1: bins 0..9
        lbp_names = [f"lbp_{i}" for i in range(10)]

        gfcc_names = [
            "gfcc_area",
            "gfcc_perimeter",
            "gfcc_major_axis_length",
            "gfcc_minor_axis_length",
            "gfcc_solidity",
            "gfcc_extent",
            "gfcc_eccentricity",
            "gfcc_circularity",
            "gfcc_axis_ratio",
            "gfcc_convex_ratio",
            "gfcc_bbox_ratio"
        ]

        return glcm_names + lbp_names + gfcc_names

    # ------------------------------------------------
    # GLCM FEATURES (MULTI-DIRECTIONAL, MULTI-SCALE)
    # ------------------------------------------------

    def glcm_features(self, img):

        # Quantize image from 256 → 64 gray levels
        img = (img / 4).astype(np.uint8)

        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        # FIX: Use multiple distances for multi-scale texture
        glcm = graycomatrix(
            img,
            distances=[1, 2, 3],
            angles=angles,
            levels=64,
            symmetric=True,
            normed=True
        )

        contrast = np.mean(graycoprops(glcm, 'contrast'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        asm = np.mean(graycoprops(glcm, 'ASM'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))

        # Average over distances and angles for scalar stats
        p = glcm.mean(axis=(2, 3))  # shape: (levels, levels)

        mean = np.mean(p)
        variance = np.var(p)
        entropy = -np.sum(p * np.log2(p + 1e-10))

        # FIX: Use correct marginal means for cluster shade/prominence
        i_idx, j_idx = np.indices(p.shape)
        mu_i = np.sum(i_idx * p)
        mu_j = np.sum(j_idx * p)

        cluster_shade = np.sum((i_idx + j_idx - mu_i - mu_j) ** 3 * p)
        cluster_prominence = np.sum((i_idx + j_idx - mu_i - mu_j) ** 4 * p)

        max_prob = np.max(p)

        return [
            contrast,
            correlation,
            energy,
            homogeneity,
            asm,
            dissimilarity,
            mean,
            variance,
            entropy,
            cluster_shade,
            cluster_prominence,
            max_prob
        ]

    # ------------------------------------------------
    # LBP FEATURES (UNIFORM)
    # ------------------------------------------------

    def lbp_features(self, img):

        radius = 1
        neighbors = 8

        # FIX: Use "uniform" method — groups patterns into 10 meaningful bins
        # for P=8, giving a compact and noise-robust descriptor
        lbp = local_binary_pattern(
            img,
            neighbors,
            radius,
            method="uniform"
        )

        # uniform LBP for P=8 produces values 0..9 (10 bins)
        n_bins = neighbors + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins)
        )

        hist = hist / (np.sum(hist) + 1e-10)

        return hist.tolist()

    # ------------------------------------------------
    # GFCC FEATURES (Corpus Callosum Geometry)
    # ------------------------------------------------

    def gfcc_features(self, img):

        try:
            thresholds = threshold_multiotsu(img, classes=3)
        except Exception:
            return [0] * 11

        segmented = np.digitize(img, bins=thresholds)

        binary = (segmented == 2).astype(np.uint8)

        props = regionprops(binary)

        if len(props) == 0:
            return [0] * 11

        # FIX: Select the LARGEST region, not necessarily the first one
        p = max(props, key=lambda x: x.area)

        area = p.area
        perimeter = p.perimeter + 1e-6
        major = p.major_axis_length
        minor = p.minor_axis_length
        solidity = p.solidity
        extent = p.extent
        eccentricity = p.eccentricity

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        axis_ratio = major / (minor + 1e-6)

        convex_ratio = area / (p.convex_area + 1e-6)

        minr, minc, maxr, maxc = p.bbox
        bbox_area = (maxr - minr) * (maxc - minc)
        bbox_ratio = area / (bbox_area + 1e-6)

        return [
            area,
            perimeter,
            major,
            minor,
            solidity,
            extent,
            eccentricity,
            circularity,
            axis_ratio,
            convex_ratio,
            bbox_ratio
        ]

    # ------------------------------------------------
    # PROCESS DATASET
    # ------------------------------------------------

    def process_split(self, split):

        dataset = []

        for label in ["autism", "control"]:

            folder = self.input_dir / split / label

            files = list(folder.glob("*.png"))

            for f in tqdm(files, desc=f"{split}-{label}"):

                img = cv2.imread(str(f), 0)

                if img is None:
                    continue

                glcm = self.glcm_features(img)
                lbp = self.lbp_features(img)
                gfcc = self.gfcc_features(img)

                features = glcm + lbp + gfcc

                dataset.append(
                    [f.stem] + features + [label]
                )

        feature_names = self.get_feature_names()

        columns = ["image_id"] + feature_names + ["label"]

        df = pd.DataFrame(dataset, columns=columns)

        df.to_csv(
            self.output_dir / f"{split}_features.csv",
            index=False
        )

    # ------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------

    def run(self):

        self.process_split("train")
        self.process_split("test")