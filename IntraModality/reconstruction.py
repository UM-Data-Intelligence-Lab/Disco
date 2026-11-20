import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(
    z_recon: torch.Tensor,
    z_target: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:


    if loss_type == "mse":
        return F.mse_loss(z_recon, z_target)
    elif loss_type == "l1":
        return F.l1_loss(z_recon, z_target)
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss(z_recon, z_target)
    elif loss_type == "cosine":

        return 1.0 - F.cosine_similarity(z_recon, z_target, dim=-1).mean()
    else:
        raise ValueError(f"Loss type: {loss_type}")


class ReconstructionModule(nn.Module):


    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        loss_type: str = "cosine",
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

        self.loss_type = loss_type

    def forward(self, x_orig: torch.Tensor, x_aug: torch.Tensor) -> dict:


        with torch.no_grad():
            z_orig = self.encoder(x_orig)


        z_aug = self.encoder(x_aug)


        z_recon = self.predictor(z_aug)


        loss_recon = reconstruction_loss(z_recon, z_orig.detach(), self.loss_type)

        return {
            "z_orig": z_orig,
            "z_aug": z_aug,
            "z_recon": z_recon,
            "loss_recon": loss_recon,
        }

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:


        self.eval()
        z = self.encoder(x)
        return self.predictor(z)
