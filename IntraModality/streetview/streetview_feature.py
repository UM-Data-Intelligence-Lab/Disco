import os
import glob
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE_SIZE = 64


class StreetviewFeatureLoader:
    def __init__(self, data_dir: str, city: str, image_size: int = IMAGE_SIZE,
                 force_recompute: bool = False, max_workers: int = 8):
        self.data_dir = data_dir
        self.city = city
        self.image_size = image_size
        self.max_workers = max_workers

        cache_img = os.path.join(data_dir, f"{city}_streetview_images_all.npy")
        cache_labels = os.path.join(data_dir, f"{city}_streetview_labels_all.npy")

        if not force_recompute and os.path.exists(cache_img) and os.path.exists(cache_labels):
            self.image_tensor = np.load(cache_img)
            self.h3_labels = list(np.load(cache_labels, allow_pickle=True))
        else:
            self.image_tensor, self.h3_labels = self._load_all_images_parallel()
            np.save(cache_img, self.image_tensor)
            np.save(cache_labels, np.array(self.h3_labels))

        self.M = self.image_tensor.shape[0]
        self.unique_h3 = list(set(self.h3_labels))
        self.N = len(self.unique_h3)

        print(f"[StreetviewLoader] {city.upper()}")

    def _load_single_image(self, fpath: str):

        h3_id = os.path.basename(os.path.dirname(fpath))
        try:
            img = Image.open(fpath).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.transpose(2, 0, 1), h3_id
        except Exception:
            return None

    def _load_all_images_parallel(self):

        img_dir = os.path.join(self.data_dir, self.city)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"{img_dir}")

        all_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            all_files.extend(glob.glob(os.path.join(img_dir, "*", ext)))

        if not all_files:
            raise FileNotFoundError(img_dir)

        total = len(all_files)

        images = []
        h3_labels = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            future_to_idx = {
                executor.submit(self._load_single_image, fp): i
                for i, fp in enumerate(all_files)
            }


            results = [None] * total
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()


        for r in results:
            if r is not None:
                images.append(r[0])
                h3_labels.append(r[1])

        return np.array(images, dtype=np.float32), h3_labels
