import torch
import torch.nn as nn

class ModalProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1, sub_dims: list = None):


        super().__init__()
        self.sub_dims = sub_dims


        if self.sub_dims is not None:

            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.randn(1, dim) * 0.02) for dim in self.sub_dims
            ])

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sub_dims is not None:

            parts = torch.split(x, self.sub_dims, dim=-1)
            masked_parts = []

            for i, part in enumerate(parts):


                is_missing = (part.abs().sum(dim=-1, keepdim=True) < 1e-6)


                mask_token = self.mask_tokens[i].expand(x.size(0), -1)


                part_filled = torch.where(is_missing, mask_token, part)
                masked_parts.append(part_filled)


            x = torch.cat(masked_parts, dim=-1)

        return self.net(x)
