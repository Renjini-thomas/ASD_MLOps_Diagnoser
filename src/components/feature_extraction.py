import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from torchvision import models, transforms


class FeatureExtraction:

    def __init__(self):

        self.input_dir = Path("data/preprocessed")
        self.output_dir = Path("artifacts/features")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------- GPU SETUP ----------------
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        torch.backends.cudnn.benchmark = True

        # ---------------- LOAD MODEL ----------------
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # remove classifier
        self.feature_extractor = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # ---------------- TRANSFORM ----------------
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.batch_size = 16   # SAFE for 4GB GPU

    # --------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------

    def load_image(self, path):

        img = cv2.imread(str(path), 0)

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = self.transform(img)

        return img

    # --------------------------------------------
    # PROCESS ONE SPLIT
    # --------------------------------------------

    def process_split(self, split):

        rows = []

        image_tensors = []
        meta = []

        for label in ["autism", "control"]:

            folder = self.input_dir / split / label
            files = list(folder.glob("*.png"))

            for f in tqdm(files, desc=f"{split}-{label}"):

                img = self.load_image(f)

                if img is None:
                    continue

                image_tensors.append(img)
                meta.append((f.stem, label))

                # ---- batch inference ----
                if len(image_tensors) == self.batch_size:

                    self.extract_batch(
                        image_tensors,
                        meta,
                        rows
                    )

                    image_tensors = []
                    meta = []

        # last batch
        if len(image_tensors) > 0:
            self.extract_batch(
                image_tensors,
                meta,
                rows
            )

        # ----- save dataframe -----
        feature_dim = len(rows[0]) - 2

        columns = (
            ["subject_id"]
            + [f"deep_{i}" for i in range(feature_dim)]
            + ["label"]
        )

        df = pd.DataFrame(rows, columns=columns)

        df.to_csv(
            self.output_dir / f"{split}_deep_features.csv",
            index=False
        )

        print(split, "shape:", df.shape)

    # --------------------------------------------
    # BATCH FEATURE EXTRACTION
    # --------------------------------------------

    def extract_batch(self, tensors, meta, rows):

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            feats = self.feature_extractor(batch)

        feats = feats.view(feats.size(0), -1).cpu().numpy()

        for i in range(len(meta)):
            subject_id, label = meta[i]
            feature_vector = feats[i].tolist()

            rows.append(
                [subject_id] + feature_vector + [label]
            )

    # --------------------------------------------

    def run(self):

        self.process_split("train")
        self.process_split("test")

        print("✅ Deep Feature Extraction Completed")
import cv2
import numpy as np
import pandas as pd
# import scipy.stats
# from pathlib import Path
# from tqdm import tqdm

# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
# from skimage.filters import threshold_multiotsu
# from skimage.measure import regionprops
# import pywt


# class FeatureExtraction:

#     def __init__(self):

#         self.input_dir = Path("data/preprocessed")
#         self.output_dir = Path("artifacts/features")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     # ------------------------------------------------
#     # FEATURE NAMES
#     # ------------------------------------------------

#     def get_feature_names(self):

#         # 12 GLCM features
#         glcm_names = [
#             "glcm_contrast",
#             "glcm_correlation",
#             "glcm_energy",
#             "glcm_homogeneity",
#             "glcm_asm",
#             "glcm_dissimilarity",
#             "glcm_mean",
#             "glcm_variance",
#             "glcm_entropy",
#             "glcm_cluster_shade",
#             "glcm_cluster_prominence",
#             "glcm_max_probability"
#         ]

#         # Multi-radius LBP: R=1 (10 bins), R=2 (18 bins), R=3 (26 bins) = 54 features
#         lbp_names = (
#             [f"lbp_r1_{i}" for i in range(10)] +
#             [f"lbp_r2_{i}" for i in range(18)] +
#             [f"lbp_r3_{i}" for i in range(26)]
#         )

#         # 11 Corpus Callosum geometry features
#         gfcc_names = [
#             "gfcc_area",
#             "gfcc_perimeter",
#             "gfcc_major_axis_length",
#             "gfcc_minor_axis_length",
#             "gfcc_solidity",
#             "gfcc_extent",
#             "gfcc_eccentricity",
#             "gfcc_circularity",
#             "gfcc_axis_ratio",
#             "gfcc_convex_ratio",
#             "gfcc_bbox_ratio"
#         ]

        
#         return (
#             glcm_names +
#             lbp_names +
#             gfcc_names 
#         )

#     # ------------------------------------------------
#     # 1. GLCM FEATURES (multi-directional, multi-scale)
#     # ------------------------------------------------

#     def glcm_features(self, img):

#         # Quantize image from 256 → 64 gray levels
#         img_q = (img / 4).astype(np.uint8)

#         angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

#         glcm = graycomatrix(
#             img_q,
#             distances=[1, 2, 3],
#             angles=angles,
#             levels=64,
#             symmetric=True,
#             normed=True
#         )

#         contrast     = np.mean(graycoprops(glcm, 'contrast'))
#         correlation  = np.mean(graycoprops(glcm, 'correlation'))
#         energy       = np.mean(graycoprops(glcm, 'energy'))
#         homogeneity  = np.mean(graycoprops(glcm, 'homogeneity'))
#         asm          = np.mean(graycoprops(glcm, 'ASM'))
#         dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))

#         # Average over distances and angles
#         p = glcm.mean(axis=(2, 3))   # shape: (levels, levels)

#         mean     = np.mean(p)
#         variance = np.var(p)
#         entropy  = -np.sum(p * np.log2(p + 1e-10))

#         i_idx, j_idx = np.indices(p.shape)
#         mu_i = np.sum(i_idx * p)
#         mu_j = np.sum(j_idx * p)

#         cluster_shade      = np.sum((i_idx + j_idx - mu_i - mu_j) ** 3 * p)
#         cluster_prominence = np.sum((i_idx + j_idx - mu_i - mu_j) ** 4 * p)
#         max_prob           = np.max(p)

#         return [
#             contrast, correlation, energy, homogeneity,
#             asm, dissimilarity, mean, variance, entropy,
#             cluster_shade, cluster_prominence, max_prob
#         ]

#     # ------------------------------------------------
#     # 2. MULTI-RADIUS LBP FEATURES
#     # ------------------------------------------------

#     def lbp_features(self, img):
#         """
#         Extract uniform LBP at 3 radii (1, 2, 3) to capture
#         texture patterns at multiple spatial scales.
#         R=1: 10 bins, R=2: 18 bins, R=3: 26 bins → 54 total
#         """
#         features = []

#         for radius, neighbors in [(1, 8), (2, 16), (3, 24)]:
#             lbp = local_binary_pattern(img, neighbors, radius, method="uniform")
#             n_bins = neighbors + 2
#             hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
#             hist = hist / (np.sum(hist) + 1e-10)
#             features.extend(hist.tolist())

#         return features

#     # ------------------------------------------------
#     # 3. GFCC FEATURES (Corpus Callosum Geometry)
#     # ------------------------------------------------

#     def gfcc_features(self, img):

#         try:
#             thresholds = threshold_multiotsu(img, classes=3)
#         except Exception:
#             return [0] * 11

#         segmented = np.digitize(img, bins=thresholds)
#         binary    = (segmented == 2).astype(np.uint8)
#         props     = regionprops(binary)

#         if len(props) == 0:
#             return [0] * 11

#         # Select the largest region
#         p = max(props, key=lambda x: x.area)

#         area         = p.area
#         perimeter    = p.perimeter + 1e-6
#         major        = p.major_axis_length
#         minor        = p.minor_axis_length
#         solidity     = p.solidity
#         extent       = p.extent
#         eccentricity = p.eccentricity

#         circularity  = (4 * np.pi * area) / (perimeter ** 2)
#         axis_ratio   = major / (minor + 1e-6)
#         convex_ratio = area / (p.convex_area + 1e-6)

#         minr, minc, maxr, maxc = p.bbox
#         bbox_area  = (maxr - minr) * (maxc - minc)
#         bbox_ratio = area / (bbox_area + 1e-6)

#         return [
#             area, perimeter, major, minor, solidity,
#             extent, eccentricity, circularity,
#             axis_ratio, convex_ratio, bbox_ratio
#         ]

    
#     # # ------------------------------------------------
#     # # PROCESS DATASET
#     # # ------------------------------------------------

#     def process_split(self, split):

#         dataset = []

#         for label in ["autism", "control"]:

#             folder = self.input_dir / split / label
#             files  = list(folder.glob("*.png"))

#             for f in tqdm(files, desc=f"{split}-{label}"):

#                 img = cv2.imread(str(f), 0)

#                 if img is None:
#                     continue

#                 try:
#                     features = (
#                         self.glcm_features(img)        +   # 12
#                         self.lbp_features(img)         +   # 54
#                         self.gfcc_features(img)        # 11    
#                     )                                      # = 128 total features
#                 except Exception as e:
#                     print(f"Error processing {f.name}: {e}")
#                     continue

#                 dataset.append([f.stem] + features + [label])

#         feature_names = self.get_feature_names()
#         columns       = ["image_id"] + feature_names + ["label"]

#         df = pd.DataFrame(dataset, columns=columns)
#         df.to_csv(self.output_dir / f"{split}_features.csv", index=False)

#         print(f"{split}: {len(df)} samples, {len(feature_names)} features extracted")

#     # ------------------------------------------------
#     # RUN PIPELINE
#     # ------------------------------------------------

#     def run(self):

#         self.process_split("train")
#         self.process_split("test")


# VERSION 32
# import cv2
# import numpy as np
# import pandas as pd
# import pywt
# import scipy.stats

# from pathlib import Path
# from tqdm import tqdm

# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# class FeatureExtraction:

#     def __init__(self):

#         self.input_dir = Path("data/preprocessed")
#         self.output_dir = Path("artifacts/features")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     # ------------------------------------------------
#     # FEATURE NAMES
#     # ------------------------------------------------

#     def get_feature_names(self):

#         glcm = [
#             "glcm_contrast",
#             "glcm_correlation",
#             "glcm_energy",
#             "glcm_homogeneity",
#             "glcm_entropy",
#             "glcm_dissimilarity"
#         ]

#         lbp = [f"lbp_{i}" for i in range(10)]

#         hist = [
#             "int_mean",
#             "int_std",
#             "int_skew",
#             "int_kurtosis",
#             "int_p10",
#             "int_p50",
#             "int_p90"
#         ]

#         wavelet = [
#             "wavelet_LL_energy",
#             "wavelet_LH_energy",
#             "wavelet_HL_energy",
#             "wavelet_HH_energy"
#         ]

#         return glcm + lbp + hist + wavelet

#     # ------------------------------------------------
#     # GLCM
#     # ------------------------------------------------

#     def glcm_features(self, img):

#         img_q = (img / 4).astype(np.uint8)

#         glcm = graycomatrix(
#             img_q,
#             distances=[1, 2],
#             angles=[0, np.pi/2],
#             levels=64,
#             symmetric=True,
#             normed=True
#         )

#         contrast = np.mean(graycoprops(glcm, 'contrast'))
#         corr = np.mean(graycoprops(glcm, 'correlation'))
#         energy = np.mean(graycoprops(glcm, 'energy'))
#         homo = np.mean(graycoprops(glcm, 'homogeneity'))
#         diss = np.mean(graycoprops(glcm, 'dissimilarity'))

#         p = glcm.mean(axis=(2,3))
#         entropy = -np.sum(p * np.log2(p + 1e-10))

#         return [contrast, corr, energy, homo, entropy, diss]

#     # ------------------------------------------------
#     # LBP (REDUCED)
#     # ------------------------------------------------

#     def lbp_features(self, img):

#         lbp = local_binary_pattern(img, 8, 1, method="uniform")
#         hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
#         hist = hist / (np.sum(hist) + 1e-10)

#         return hist.tolist()

#     # ------------------------------------------------
#     # HISTOGRAM FEATURES
#     # ------------------------------------------------

#     def histogram_features(self, img):

#         flat = img.flatten()

#         mean = np.mean(flat)
#         std = np.std(flat)
#         skew = scipy.stats.skew(flat)
#         kurt = scipy.stats.kurtosis(flat)

#         p10 = np.percentile(flat, 10)
#         p50 = np.percentile(flat, 50)
#         p90 = np.percentile(flat, 90)

#         return [mean, std, skew, kurt, p10, p50, p90]

#     # ------------------------------------------------
#     # WAVELET FEATURES
#     # ------------------------------------------------

#     def wavelet_features(self, img):

#         coeffs = pywt.dwt2(img, 'haar')

#         LL, (LH, HL, HH) = coeffs

#         def energy(x):
#             return np.sum(x ** 2) / (x.size + 1e-10)

#         return [
#             energy(LL),
#             energy(LH),
#             energy(HL),
#             energy(HH)
#         ]

#     # ------------------------------------------------
#     # PROCESS SPLIT
#     # ------------------------------------------------

#     def process_split(self, split):

#         dataset = []

#         for label in ["autism", "control"]:

#             folder = self.input_dir / split / label
#             files = list(folder.glob("*.png"))

#             for f in tqdm(files):

#                 img = cv2.imread(str(f), 0)

#                 if img is None:
#                     continue

#                 features = (
#                     self.glcm_features(img) +
#                     self.lbp_features(img) +
#                     self.histogram_features(img) +
#                     self.wavelet_features(img)
#                 )

#                 dataset.append([f.stem] + features + [label])

#         columns = ["subject_id"] + self.get_feature_names() + ["label"]

#         df = pd.DataFrame(dataset, columns=columns)

#         df.to_csv(self.output_dir / f"{split}_features.csv", index=False)

#         print(split, "shape:", df.shape)

#     # ------------------------------------------------

#     def run(self):

#         self.process_split("train")
#         self.process_split("test")

#         print("Axial Feature Extraction Completed")

# import cv2
# import numpy as np
# import pandas as pd
# import pywt
# import scipy.stats

# from pathlib import Path
# from tqdm import tqdm

# from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# class FeatureExtraction:

#     def __init__(self):

#         self.input_dir = Path("data/preprocessed")
#         self.output_dir = Path("artifacts/features")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#     # ------------------------------------------------
#     # FEATURE NAMES
#     # ------------------------------------------------

#     def get_feature_names(self):

#         glcm = [
#             "glcm_contrast",
#             "glcm_correlation",
#             "glcm_energy",
#             "glcm_homogeneity",
#             "glcm_entropy",
#             "glcm_dissimilarity"
#         ]

#         lbp = [f"lbp_{i}" for i in range(10)]

#         hist = [
#             "int_mean",
#             "int_std",
#             "int_skew",
#             "int_kurtosis",
#             "int_p10",
#             "int_p50",
#             "int_p90"
#         ]

#         wavelet = [
#             "wavelet_LL_energy",
#             "wavelet_LH_energy",
#             "wavelet_HL_energy",
#             "wavelet_HH_energy"
#         ]

#         return glcm + lbp + hist + wavelet

#     # ------------------------------------------------
#     # GLCM
#     # ------------------------------------------------

#     def glcm_features(self, img):

#         img_q = (img / 4).astype(np.uint8)

#         glcm = graycomatrix(
#             img_q,
#             distances=[1, 2],
#             angles=[0, np.pi/2],
#             levels=64,
#             symmetric=True,
#             normed=True
#         )

#         contrast = np.mean(graycoprops(glcm, 'contrast'))
#         corr = np.mean(graycoprops(glcm, 'correlation'))
#         energy = np.mean(graycoprops(glcm, 'energy'))
#         homo = np.mean(graycoprops(glcm, 'homogeneity'))
#         diss = np.mean(graycoprops(glcm, 'dissimilarity'))

#         p = glcm.mean(axis=(2, 3))
#         entropy = -np.sum(p * np.log2(p + 1e-10))

#         return [contrast, corr, energy, homo, entropy, diss]

#     # ------------------------------------------------
#     # LBP
#     # ------------------------------------------------

#     def lbp_features(self, img):

#         lbp = local_binary_pattern(img, 8, 1, method="uniform")
#         hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
#         hist = hist / (np.sum(hist) + 1e-10)

#         return hist.tolist()

#     # ------------------------------------------------
#     # HISTOGRAM
#     # ------------------------------------------------

#     def histogram_features(self, img):

#         flat = img.flatten()

#         mean = np.mean(flat)
#         std = np.std(flat)
#         skew = scipy.stats.skew(flat)
#         kurt = scipy.stats.kurtosis(flat)

#         p10 = np.percentile(flat, 10)
#         p50 = np.percentile(flat, 50)
#         p90 = np.percentile(flat, 90)

#         return [mean, std, skew, kurt, p10, p50, p90]

#     # ------------------------------------------------
#     # WAVELET
#     # ------------------------------------------------

#     def wavelet_features(self, img):

#         coeffs = pywt.dwt2(img, 'haar')

#         LL, (LH, HL, HH) = coeffs

#         def energy(x):
#             return np.sum(x ** 2) / (x.size + 1e-10)

#         return [
#             energy(LL),
#             energy(LH),
#             energy(HL),
#             energy(HH)
#         ]

#     # ------------------------------------------------
#     # PROCESS SPLIT (3-SLICE CONCATENATION)
#     # ------------------------------------------------

#     def process_split(self, split):

#         dataset = []

#         for label in ["autism", "control"]:

#             folder = self.input_dir / split / label
#             files = list(folder.glob("*.png"))

#             # ⭐ group slices per subject
#             subject_groups = {}

#             for f in files:
#                 subject_id = f.stem.rsplit("_", 1)[0]
#                 subject_groups.setdefault(subject_id, []).append(f)

#             # ⭐ process each subject
#             for subject_id, slice_files in tqdm(subject_groups.items()):

#                 slice_files = sorted(slice_files)

#                 concat_features = []

#                 for f in slice_files:

#                     img = cv2.imread(str(f), 0)

#                     if img is None:
#                         continue

#                     features = (
#                         self.glcm_features(img) +
#                         self.lbp_features(img) +
#                         self.histogram_features(img) +
#                         self.wavelet_features(img)
#                     )

#                     concat_features.extend(features)

#                 if len(concat_features) == 0:
#                     continue

#                 dataset.append([subject_id] + concat_features + [label])

#         # ⭐ build column names
#         base_names = self.get_feature_names()

#         columns = ["subject_id"]

#         for tag in ["m1", "m0", "p1"]:
#             columns += [f"{name}_{tag}" for name in base_names]

#         columns += ["label"]

#         df = pd.DataFrame(dataset, columns=columns)

#         df.to_csv(self.output_dir / f"{split}_features.csv", index=False)

#         print(split, "shape:", df.shape)

#     # ------------------------------------------------

#     def run(self):

#         self.process_split("train")
#         self.process_split("test")

#         print("3-Slice Feature Concatenation Completed")