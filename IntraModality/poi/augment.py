import numpy as np


class POIAugmentor:


    def __init__(self, poi_matrix: np.ndarray, corrupt_ratio: float = 0.23):


        self.original = poi_matrix.copy()
        self.N, self.C = poi_matrix.shape
        self.num_corrupt = max(1, round(self.C * corrupt_ratio))
        self.max_val = max(int(poi_matrix.max()), 1)


    def augment_replace(self) -> np.ndarray:


        aug = self.original.copy()
        for i in range(self.N):
            cols = np.random.choice(self.C, self.num_corrupt, replace=False)
            aug[i, cols] = np.random.randint(0, self.max_val + 1, size=self.num_corrupt)
        return aug


    def augment_insert(self) -> np.ndarray:


        aug = self.original.copy()
        for i in range(self.N):
            positions = sorted(
                np.random.choice(self.C, self.num_corrupt, replace=False),
                reverse=True,
            )
            for pos in positions:
                random_val = np.random.randint(0, self.max_val + 1)

                aug[i, pos + 1 :] = aug[i, pos:-1].copy()
                aug[i, pos] = random_val
        return aug


    def augment_delete(self) -> np.ndarray:


        aug = self.original.copy()
        for i in range(self.N):
            cols = np.random.choice(self.C, self.num_corrupt, replace=False)
            aug[i, cols] = 0
        return aug


    def build_all(self) -> dict:


        return {
            "replace": self.augment_replace(),
            "insert": self.augment_insert(),
            "delete": self.augment_delete(),
        }
