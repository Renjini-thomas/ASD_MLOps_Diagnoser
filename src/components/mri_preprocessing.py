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
        img = img / np.max(img)

        img = (img * 255).astype(np.uint8)

        return img

    def process_file(self, mgz_path, save_path):

        img = nib.load(str(mgz_path)).get_fdata()

        # Mid sagittal slice
        mid_index = img.shape[0] // 2
        slice_img = img[mid_index, :, :]

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

                files = list(input_folder.glob("*.mgz"))

                for file in tqdm(files):

                    save_path = output_folder / (file.stem + ".png")

                    self.process_file(file, save_path)