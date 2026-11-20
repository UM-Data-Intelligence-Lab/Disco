import torch
import torch.nn as nn

class ModalitySpecificProjector(nn.Module):


    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()


        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        return self.net(x)
