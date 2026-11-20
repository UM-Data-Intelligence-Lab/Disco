import os
import numpy as np
import torch


class FlowFeatureLoader:


    def __init__(self, data_dir: str, city: str, norm: str = "log1p_zscore"):


        self.data_dir = data_dir
        self.city = city
        self.norm = norm


        profile_path = os.path.join(data_dir, f"{city}_flow_h3_profile.npy")
        h3_list_path = os.path.join(data_dir, f"{city}_flow_h3_list.npy")

        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"{profile_path}")
        if not os.path.exists(h3_list_path):
            raise FileNotFoundError(f"{h3_list_path}")

        self.raw_matrix = np.load(profile_path).astype(np.float32)
        self.h3_list = np.load(h3_list_path, allow_pickle=True)

        self.N = self.raw_matrix.shape[0]
        self.D = self.raw_matrix.shape[1]



        self.h3_to_idx = {h: i for i, h in enumerate(self.h3_list)}


        self.feature_matrix = self._normalize(self.raw_matrix.copy())


        sparsity = (self.raw_matrix == 0).mean()
        print(f"[FlowFeature] {city.upper()} | shape={self.raw_matrix.shape} | sparsity={sparsity:.2%}")


    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.norm == "none":
            return X

        if self.norm in ("log1p", "log1p_zscore"):
            X = np.log1p(X)

        if self.norm in ("zscore", "log1p_zscore"):
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-8
            X = (X - mean) / std

        elif self.norm == "minmax":
            mn = X.min(axis=0, keepdims=True)
            mx = X.max(axis=0, keepdims=True) + 1e-8
            X = (X - mn) / (mx - mn)

        return X


    def get_numpy(self) -> np.ndarray:

        return self.feature_matrix

    def get_tensor(self, device: str = "cpu") -> torch.Tensor:

        return torch.from_numpy(self.feature_matrix).to(device)

    def get_h3_list(self) -> list:

        return list(self.h3_list)

    def get_h3_to_idx(self) -> dict:

        return dict(self.h3_to_idx)

    def get_raw_numpy(self) -> np.ndarray:

        return self.raw_matrix

    def get_inflow_outflow(self):


        half = self.N * 24
        return self.raw_matrix[:, :half], self.raw_matrix[:, half:]

    def __len__(self):
        return self.N

    def __repr__(self):
        return (f"FlowFeatureLoader(city={self.city}, N={self.N}, D={self.D}, "
                f"norm={self.norm})")
