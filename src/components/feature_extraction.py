import cv2
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path
from tqdm import tqdm

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops
import pywt


class FeatureExtraction:

    def __init__(self):

        self.input_dir = Path("data/preprocessed")
        self.output_dir = Path("artifacts/features")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # FEATURE NAMES
    # ------------------------------------------------

    def get_feature_names(self):

        # 12 GLCM features
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

        # Multi-radius LBP: R=1 (10 bins), R=2 (18 bins), R=3 (26 bins) = 54 features
        lbp_names = (
            [f"lbp_r1_{i}" for i in range(10)] +
            [f"lbp_r2_{i}" for i in range(18)] +
            [f"lbp_r3_{i}" for i in range(26)]
        )

        # 11 Corpus Callosum geometry features
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

        # 20 Wavelet features (4 bands × 5 stats)
        bands = ["LL", "LH", "HL", "HH"]
        stats = ["mean", "std", "var", "max", "min"]
        wavelet_names = [f"wavelet_{b}_{s}" for b in bands for s in stats]

        # 6 Asymmetry features (expanded from 2)
        asymmetry_names = [
            "asym_mean_diff",
            "asym_std_diff",
            "asym_energy",
            "asym_bilateral_corr",
            "asym_iqr",
            "asym_high_frac"
        ]

        # 5 HOG summary features
        hog_names = [
            "hog_mean",
            "hog_std",
            "hog_energy",
            "hog_p75",
            "hog_max"
        ]

        # 9 Intensity distribution features
        intensity_names = [
            "intensity_p10",
            "intensity_p25",
            "intensity_p50",
            "intensity_p75",
            "intensity_p90",
            "intensity_skew",
            "intensity_kurtosis",
            "intensity_iqr",
            "intensity_above_mean"
        ]

        # 11 CC sub-region features (5 regions × 2 stats + genu/splenium ratio)
        cc_sub_names = (
            [f"cc_region_{i}_{t}" for i in range(5)
             for t in ["area_ratio", "thickness"]] +
            ["cc_genu_splenium_ratio"]
        )

        return (
            glcm_names +
            lbp_names +
            gfcc_names +
            wavelet_names +
            asymmetry_names +
            hog_names +
            intensity_names +
            cc_sub_names
        )

    # ------------------------------------------------
    # 1. GLCM FEATURES (multi-directional, multi-scale)
    # ------------------------------------------------

    def glcm_features(self, img):

        # Quantize image from 256 → 64 gray levels
        img_q = (img / 4).astype(np.uint8)

        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(
            img_q,
            distances=[1, 2, 3],
            angles=angles,
            levels=64,
            symmetric=True,
            normed=True
        )

        contrast     = np.mean(graycoprops(glcm, 'contrast'))
        correlation  = np.mean(graycoprops(glcm, 'correlation'))
        energy       = np.mean(graycoprops(glcm, 'energy'))
        homogeneity  = np.mean(graycoprops(glcm, 'homogeneity'))
        asm          = np.mean(graycoprops(glcm, 'ASM'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))

        # Average over distances and angles
        p = glcm.mean(axis=(2, 3))   # shape: (levels, levels)

        mean     = np.mean(p)
        variance = np.var(p)
        entropy  = -np.sum(p * np.log2(p + 1e-10))

        i_idx, j_idx = np.indices(p.shape)
        mu_i = np.sum(i_idx * p)
        mu_j = np.sum(j_idx * p)

        cluster_shade      = np.sum((i_idx + j_idx - mu_i - mu_j) ** 3 * p)
        cluster_prominence = np.sum((i_idx + j_idx - mu_i - mu_j) ** 4 * p)
        max_prob           = np.max(p)

        return [
            contrast, correlation, energy, homogeneity,
            asm, dissimilarity, mean, variance, entropy,
            cluster_shade, cluster_prominence, max_prob
        ]

    # ------------------------------------------------
    # 2. MULTI-RADIUS LBP FEATURES
    # ------------------------------------------------

    def lbp_features(self, img):
        """
        Extract uniform LBP at 3 radii (1, 2, 3) to capture
        texture patterns at multiple spatial scales.
        R=1: 10 bins, R=2: 18 bins, R=3: 26 bins → 54 total
        """
        features = []

        for radius, neighbors in [(1, 8), (2, 16), (3, 24)]:
            lbp = local_binary_pattern(img, neighbors, radius, method="uniform")
            n_bins = neighbors + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist / (np.sum(hist) + 1e-10)
            features.extend(hist.tolist())

        return features

    # ------------------------------------------------
    # 3. GFCC FEATURES (Corpus Callosum Geometry)
    # ------------------------------------------------

    def gfcc_features(self, img):

        try:
            thresholds = threshold_multiotsu(img, classes=3)
        except Exception:
            return [0] * 11

        segmented = np.digitize(img, bins=thresholds)
        binary    = (segmented == 2).astype(np.uint8)
        props     = regionprops(binary)

        if len(props) == 0:
            return [0] * 11

        # Select the largest region
        p = max(props, key=lambda x: x.area)

        area         = p.area
        perimeter    = p.perimeter + 1e-6
        major        = p.major_axis_length
        minor        = p.minor_axis_length
        solidity     = p.solidity
        extent       = p.extent
        eccentricity = p.eccentricity

        circularity  = (4 * np.pi * area) / (perimeter ** 2)
        axis_ratio   = major / (minor + 1e-6)
        convex_ratio = area / (p.convex_area + 1e-6)

        minr, minc, maxr, maxc = p.bbox
        bbox_area  = (maxr - minr) * (maxc - minc)
        bbox_ratio = area / (bbox_area + 1e-6)

        return [
            area, perimeter, major, minor, solidity,
            extent, eccentricity, circularity,
            axis_ratio, convex_ratio, bbox_ratio
        ]

    # ------------------------------------------------
    # 4. WAVELET FEATURES
    # ------------------------------------------------

    def wavelet_features(self, img):

        coeffs = pywt.dwt2(img.astype(float), 'haar')
        LL, (LH, HL, HH) = coeffs

        features = []
        for band in [LL, LH, HL, HH]:
            features.append(np.mean(band))
            features.append(np.std(band))
            features.append(np.var(band))
            features.append(np.max(band))
            features.append(np.min(band))

        return features

    # ------------------------------------------------
    # 5. ASYMMETRY FEATURES (expanded — 6 features)
    # ------------------------------------------------

    def asymmetry_features(self, img):
        """
        Bilateral hemisphere asymmetry features.
        ASD brains show measurable structural asymmetry differences.
        """
        h, w   = img.shape
        mid    = w // 2
        left   = img[:, :mid].astype(float)
        right  = np.fliplr(img[:, mid:]).astype(float)

        # Ensure equal width in case of odd image width
        min_w = min(left.shape[1], right.shape[1])
        left  = left[:, :min_w]
        right = right[:, :min_w]

        diff = np.abs(left - right)

        mean_diff        = np.mean(diff)
        std_diff         = np.std(diff)
        energy           = np.sum(diff) / (h * min_w + 1e-6)
        bilateral_corr   = np.corrcoef(left.ravel(), right.ravel())[0, 1]
        iqr              = float(np.percentile(diff, 75) - np.percentile(diff, 25))
        high_frac        = float(np.mean(diff > np.mean(diff)))

        return [mean_diff, std_diff, energy, bilateral_corr, iqr, high_frac]

    # ------------------------------------------------
    # 6. HOG FEATURES (structural gradient patterns)
    # ------------------------------------------------

    def hog_features(self, img):
        """
        Histogram of Oriented Gradients captures edge/gradient
        structure — useful for detecting morphological differences
        in brain tissue organisation.
        """
        hog_vec = hog(
            img,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        return [
            float(np.mean(hog_vec)),
            float(np.std(hog_vec)),
            float(np.sum(hog_vec ** 2)),        # HOG energy
            float(np.percentile(hog_vec, 75)),
            float(np.max(hog_vec))
        ]

    # ------------------------------------------------
    # 7. INTENSITY DISTRIBUTION FEATURES
    # ------------------------------------------------

    def intensity_features(self, img):
        """
        Percentile-based intensity statistics capture shifts in
        brain tissue density distribution between ASD and control.
        """
        flat = img.ravel().astype(float)

        p10 = float(np.percentile(flat, 10))
        p25 = float(np.percentile(flat, 25))
        p50 = float(np.percentile(flat, 50))
        p75 = float(np.percentile(flat, 75))
        p90 = float(np.percentile(flat, 90))

        skewness = float(scipy.stats.skew(flat))
        kurtosis = float(scipy.stats.kurtosis(flat))
        iqr      = p75 - p25
        above_mean = float(np.mean(flat > np.mean(flat)))

        return [p10, p25, p50, p75, p90, skewness, kurtosis, iqr, above_mean]

    # ------------------------------------------------
    # 8. CC SUB-REGION FEATURES (clinically validated)
    # ------------------------------------------------

    def cc_subregion_features(self, img):
        """
        Divides the corpus callosum into 5 anatomical sub-regions:
        genu, rostral body, anterior midbody, posterior midbody, splenium.
        Genu/splenium size differences are a clinically validated ASD marker.
        Returns 11 features: 5 regions × (area_ratio + thickness) + genu/splenium ratio.
        """
        try:
            thresholds = threshold_multiotsu(img, classes=3)
        except Exception:
            return [0] * 11

        binary = (np.digitize(img, bins=thresholds) == 2).astype(np.uint8)

        if binary.sum() == 0:
            return [0] * 11

        # Find column extent of CC mask
        cols = np.where(binary.any(axis=0))[0]

        if len(cols) < 5:
            return [0] * 11

        col_min      = int(cols[0])
        col_max      = int(cols[-1])
        region_width = max((col_max - col_min) // 5, 1)
        total_area   = binary.sum() + 1e-6

        features = []
        region_areas = []

        for i in range(5):
            c_start = col_min + i * region_width
            c_end   = c_start + region_width if i < 4 else col_max
            region  = binary[:, c_start:c_end]

            area      = float(region.sum())
            col_sums  = np.sum(region, axis=0)
            thickness = float(np.mean(col_sums[col_sums > 0])) if col_sums.any() else 0.0

            features.append(area / total_area)
            features.append(thickness)
            region_areas.append(area)

        # Genu (region 0) / Splenium (region 4) ratio — key ASD biomarker
        genu_splenium_ratio = region_areas[0] / (region_areas[4] + 1e-6)
        features.append(float(genu_splenium_ratio))

        return features

    # ------------------------------------------------
    # PROCESS DATASET
    # ------------------------------------------------

    def process_split(self, split):

        dataset = []

        for label in ["autism", "control"]:

            folder = self.input_dir / split / label
            files  = list(folder.glob("*.png"))

            for f in tqdm(files, desc=f"{split}-{label}"):

                img = cv2.imread(str(f), 0)

                if img is None:
                    continue

                try:
                    features = (
                        self.glcm_features(img)        +   # 12
                        self.lbp_features(img)         +   # 54
                        self.gfcc_features(img)        +   # 11
                        self.wavelet_features(img)     +   # 20
                        self.asymmetry_features(img)   +   #  6
                        self.hog_features(img)         +   #  5
                        self.intensity_features(img)   +   #  9
                        self.cc_subregion_features(img)    # 11
                    )                                      # = 128 total features
                except Exception as e:
                    print(f"Error processing {f.name}: {e}")
                    continue

                dataset.append([f.stem] + features + [label])

        feature_names = self.get_feature_names()
        columns       = ["image_id"] + feature_names + ["label"]

        df = pd.DataFrame(dataset, columns=columns)
        df.to_csv(self.output_dir / f"{split}_features.csv", index=False)

        print(f"{split}: {len(df)} samples, {len(feature_names)} features extracted")

    # ------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------

    def run(self):

        self.process_split("train")
        self.process_split("test")