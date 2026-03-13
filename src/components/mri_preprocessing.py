import nibabel as nib
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


class MRIPreprocessing:

    def __init__(self):

        self.input_dir = Path("data/processed")
        self.output_dir = Path("data/preprocessed")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize(self, img):

        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)

        img = (img * 255).astype(np.uint8)

        return img

    def process_file(self, mgz_path, save_path):

        img = nib.load(str(mgz_path)).get_fdata()

        # # Mid sagittal slice
        # mid_index = img.shape[0] // 2
        # slice_img = img[mid_index, :, :]
        # mid axial slice
        mid_index = img.shape[2] // 2
        slice_img = img[:, :, mid_index]

        slice_img = self.normalize(slice_img)

        # Resize
        slice_img = cv2.resize(slice_img, (256, 256))

        cv2.imwrite(str(save_path), slice_img)

    def run(self):

        for split in ["train", "test"]:

            for label in ["autism", "control"]:

                input_folder = self.input_dir / split / label
                output_folder = self.output_dir / split / label

                output_folder.mkdir(parents=True, exist_ok=True)

                subjects = list(input_folder.iterdir())

                for subject in tqdm(subjects):

                    mgz_path = subject / "mri" / "brain.mgz"

                    if not mgz_path.exists():
                        continue

                    save_path = output_folder / (subject.name + ".png")

                    self.process_file(mgz_path, save_path)
# # import nibabel as nib
# # import numpy as np
# # import cv2
# # from pathlib import Path
# # from tqdm import tqdm


# # class MRIPreprocessing:

# #     def __init__(self):

# #         # Input MGZ dataset
# #         self.input_dir = Path("data/processed")

# #         # Output PNG slices
# #         self.output_dir = Path("data/preprocessed")

# #         self.output_dir.mkdir(parents=True, exist_ok=True)

# #         # Number of sagittal slices around midline
# #         self.slice_offsets = [-2, -1, 0, 1, 2]

# #     # ----------------------------------------
# #     # NORMALIZATION
# #     # ----------------------------------------

# #     def normalize(self, img):

# #         img = img - np.min(img)

# #         max_val = np.max(img)

# #         if max_val != 0:
# #             img = img / max_val

# #         img = (img * 255).astype(np.uint8)

# #         return img

# #     # ----------------------------------------
# #     # PROCESS SINGLE MRI VOLUME
# #     # ----------------------------------------

# #     def process_file(self, mgz_path, save_folder):

# #         # Load MRI volume
# #         volume = nib.load(str(mgz_path)).get_fdata()

# #         mid_index = volume.shape[0] // 2

# #         for offset in self.slice_offsets:

# #             slice_index = mid_index + offset

# #             # Safety check
# #             if slice_index < 0 or slice_index >= volume.shape[0]:
# #                 continue

# #             slice_img = volume[slice_index, :, :]

# #             # Normalize
# #             slice_img = self.normalize(slice_img)

# #             # Resize
# #             slice_img = cv2.resize(slice_img, (256, 256))

# #             # Save filename
# #             filename = f"{mgz_path.stem}_s{offset}.png"

# #             save_path = save_folder / filename

# #             cv2.imwrite(str(save_path), slice_img)

# #     # ----------------------------------------
# #     # MAIN PIPELINE
# #     # ----------------------------------------

# #     def run(self):

# #         for split in ["train", "test"]:

# #             for label in ["autism", "control"]:

# #                 input_folder = self.input_dir / split / label
# #                 output_folder = self.output_dir / split / label

# #                 output_folder.mkdir(parents=True, exist_ok=True)

# #                 files = list(input_folder.glob("*.mgz"))

# #                 print(f"\nProcessing {split}/{label} : {len(files)} subjects")

# #                 for file in tqdm(files):

# #                     self.process_file(file, output_folder)

# import nibabel as nib
# import numpy as np
# import cv2
# from pathlib import Path
# from tqdm import tqdm


# class MRIPreprocessing:

#     def __init__(self):

#         self.input_dir = Path("data/processed")
#         self.output_dir = Path("data/preprocessed")

#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         # take 3 axial slices
#         self.slice_offsets = [-1, 0, 1]

#     # ----------------------------------------
#     # NORMALIZATION
#     # ----------------------------------------

#     def normalize(self, img):

#         img = img - np.min(img)

#         max_val = np.max(img)

#         if max_val != 0:
#             img = img / max_val

#         img = (img * 255).astype(np.uint8)

#         return img

#     # ----------------------------------------
#     # FIND BEST AXIAL SLICE
#     # ----------------------------------------

#     def get_best_axial_index(self, volume):

#         max_area = 0
#         best_index = volume.shape[2] // 2   # fallback

#         for i in range(volume.shape[2]):

#             sl = volume[:, :, i]

#             area = np.sum(sl > 0)

#             if area > max_area:
#                 max_area = area
#                 best_index = i

#         return best_index

#     # ----------------------------------------
#     # PROCESS SINGLE MRI
#     # ----------------------------------------

#     def process_file(self, mgz_path, save_folder):

#         volume = nib.load(str(mgz_path)).get_fdata()

#         best_index = self.get_best_axial_index(volume)

#         for offset in self.slice_offsets:

#             idx = best_index + offset

#             if idx < 0 or idx >= volume.shape[2]:
#                 continue

#             slice_img = volume[:, :, idx]

#             slice_img = self.normalize(slice_img)

#             slice_img = cv2.resize(slice_img, (256, 256))

#             filename = f"{mgz_path.parent.parent.name}_z{offset}.png"

#             save_path = save_folder / filename

#             cv2.imwrite(str(save_path), slice_img)

#     # ----------------------------------------
#     # MAIN PIPELINE
#     # ----------------------------------------

#     def run(self):

#         for split in ["train", "test"]:

#             for label in ["autism", "control"]:

#                 input_folder = self.input_dir / split / label
#                 output_folder = self.output_dir / split / label

#                 output_folder.mkdir(parents=True, exist_ok=True)

#                 subjects = list(input_folder.iterdir())

#                 print(f"\nProcessing {split}/{label}: {len(subjects)}")

#                 for subject in tqdm(subjects):

#                     mgz_path = subject / "mri" / "brain.mgz"

#                     if not mgz_path.exists():
#                         continue

#                     self.process_file(mgz_path, output_folder)