import torch
import torch.nn as nn

class GlobalReconstructor(nn.Module):


    def __init__(self, shared_dim: int, specific_dim: int, target_dim: int, hidden_dim: int = 512):
        super().__init__()
        concat_dim = shared_dim + specific_dim

        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, z_shared: torch.Tensor, z_specific: torch.Tensor) -> torch.Tensor:

        x_concat = torch.cat([z_shared, z_specific], dim=-1)
        return self.net(x_concat)
