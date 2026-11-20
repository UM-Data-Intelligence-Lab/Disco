import os
import glob
import numpy as np
from PIL import Image

IMAGE_SIZE = 64

class SatelliteFeatureLoader:


    def __init__(self, data_dir: str, city: str, image_size: int = IMAGE_SIZE,
                 force_recompute: bool = False):
        self.data_dir = data_dir
        self.city = city
        self.image_size = image_size

        cache_img = os.path.join(data_dir, f"{city}_satellite_images.npy")
        cache_labels = os.path.join(data_dir, f"{city}_satellite_labels.npy")

        if not force_recompute and os.path.exists(cache_img) and os.path.exists(cache_labels):
            self.image_tensor = np.load(cache_img)
            self.h3_labels = list(np.load(cache_labels, allow_pickle=True))
        else:
            self.image_tensor, self.h3_labels = self._load_all_images()
            np.save(cache_img, self.image_tensor)
            np.save(cache_labels, np.array(self.h3_labels))

        self.N = self.image_tensor.shape[0]

        print(f"[SatelliteLoader] {city.upper()}")

    def _load_all_images(self):
        img_dir = os.path.join(self.data_dir, f"{self.city}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"{img_dir}")

        all_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            all_files.extend(glob.glob(os.path.join(img_dir, ext)))

        if not all_files:
            raise FileNotFoundError(img_dir)

        images = []
        h3_labels = []

        for fpath in all_files:

            fname = os.path.basename(fpath)
            h3_id = os.path.splitext(fname)[0]

            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr.transpose(2, 0, 1))
                h3_labels.append(h3_id)
            except Exception:
                continue

        return np.array(images, dtype=np.float32), h3_labels

    def get_numpy(self) -> np.ndarray:
        return self.image_tensor

    def get_h3_labels(self) -> list:
        return list(self.h3_labels)
