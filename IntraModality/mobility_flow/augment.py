import numpy as np


class FlowAugmentor:


    def __init__(
        self,
        flow_matrix: np.ndarray,
        noise_scales: list = None,
        seed: int = None,
    ):
        self.flow_matrix = flow_matrix.astype(np.float32).copy()
        self.noise_scales = noise_scales or [0.05, 0.10, 0.20]
        self.global_std = float(self.flow_matrix.std()) + 1e-8

        if seed is not None:
            np.random.seed(seed)

    def _add_gaussian(self, scale: float) -> np.ndarray:

        sigma = self.global_std * scale
        noise = np.random.randn(*self.flow_matrix.shape).astype(np.float32) * sigma
        return self.flow_matrix + noise


    def build_all(self) -> dict:


        names = ["gauss_low", "gauss_mid", "gauss_high"]
        result = {}
        for name, scale in zip(names, self.noise_scales):
            result[name] = self._add_gaussian(scale)
        return result
