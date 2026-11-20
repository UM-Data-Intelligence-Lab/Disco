import torch
import torch.nn as nn

class LatentPredictor(nn.Module):


    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:


        return self.net(z)
