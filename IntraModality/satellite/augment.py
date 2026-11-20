import numpy as np

class SatelliteAugmentor:


    def __init__(self, image_tensor: np.ndarray, seed: int = None):

        self.image_tensor = image_tensor.astype(np.float32).copy()

        if seed is not None:
            np.random.seed(seed)


    def build_all(self) -> dict:


        result = {}


        result["rot_90"] = np.rot90(self.image_tensor, k=1, axes=(2, 3)).copy()
        result["rot_180"] = np.rot90(self.image_tensor, k=2, axes=(2, 3)).copy()
        result["rot_270"] = np.rot90(self.image_tensor, k=3, axes=(2, 3)).copy()


        return result
