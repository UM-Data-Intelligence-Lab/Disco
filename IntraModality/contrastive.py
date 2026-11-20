import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):


    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


def info_nce_loss(
    z_orig: torch.Tensor,
    z_aug: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:


    B = z_orig.size(0)
    z = torch.cat([z_orig, z_aug], dim=0)
    sim = torch.mm(z, z.t()) / temperature


    labels = torch.cat([
        torch.arange(B) + B,
        torch.arange(B),
    ], dim=0).to(z.device)


    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    return F.cross_entropy(sim, labels)


class ContrastiveModule(nn.Module):


    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        proj_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = encoder
        self.proj_head = ProjectionHead(latent_dim, proj_dim)
        self.temperature = temperature

    def forward(self, x_orig: torch.Tensor, x_aug: torch.Tensor) -> dict:


        z_orig = self.encoder(x_orig)
        z_aug = self.encoder(x_aug)

        p_orig = self.proj_head(z_orig)
        p_aug = self.proj_head(z_aug)

        loss_cl = info_nce_loss(p_orig, p_aug, self.temperature)

        return {
            "z_orig": z_orig,
            "z_aug": z_aug,
            "loss_cl": loss_cl,
        }
