import torch
import torch.nn as nn
from ..specific.loss import orthogonal_loss, masked_reconstruction_loss
from ..specific.projector import ModalitySpecificProjector


class MaskedModalityBranch(nn.Module):


    def __init__(self, input_dim: int, specific_dim: int, hidden_dim: int = 256, mask_ratio: float = 0.25):
        super().__init__()
        self.mask_ratio = mask_ratio


        self.mask_token = nn.Parameter(torch.randn(1, input_dim) * 0.02)


        self.projector = ModalitySpecificProjector(input_dim, hidden_dim, specific_dim)


        self.recon_head = nn.Linear(specific_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        use_mask: bool = True,
        compute_recon: bool = True,
    ):
        B, D = x.shape
        device = x.device

        if self.training and use_mask:

            rand_tensor = torch.rand((B, D), device=device)
            mask = rand_tensor < self.mask_ratio


            mask_tokens = self.mask_token.expand(B, -1)
            x_masked = torch.where(mask, mask_tokens, x)
        else:

            x_masked = x
            mask = torch.zeros((B, D), dtype=torch.bool, device=device)


        z_specific = self.projector(x_masked)


        x_recon = self.recon_head(z_specific) if compute_recon else None

        return z_specific, x_recon, mask


class ModalitySpecificExtractor(nn.Module):


    def __init__(self,
                 functional_dim: int = 256,
                 visual_dim: int = 256,
                 specific_dim: int = 256,
                 mask_ratio: float = 0.25):
        super().__init__()


        self.functional_branch = MaskedModalityBranch(functional_dim, specific_dim, mask_ratio=mask_ratio)


        self.visual_branch = MaskedModalityBranch(visual_dim, specific_dim, mask_ratio=mask_ratio)

    def forward(
        self,
        functional_feats: torch.Tensor,
        visual_feats: torch.Tensor,
        shared_emb_functional: torch.Tensor,
        shared_emb_visual: torch.Tensor,
        use_mask: bool = True,
    ):


        z_specific_functional, recon_functional, mask_functional = self.functional_branch(
            functional_feats,
            use_mask=use_mask,
            compute_recon=use_mask,
        )
        z_specific_visual, recon_visual, mask_visual = self.visual_branch(
            visual_feats,
            use_mask=use_mask,
            compute_recon=use_mask,
        )


        if use_mask:
            loss_recon_functional = masked_reconstruction_loss(
                recon_functional,
                functional_feats,
                mask_functional,
            )
            loss_recon_visual = masked_reconstruction_loss(
                recon_visual,
                visual_feats,
                mask_visual,
            )
        else:
            loss_recon_functional = functional_feats.new_zeros(())
            loss_recon_visual = visual_feats.new_zeros(())


        loss_ortho_functional = orthogonal_loss(z_specific_functional, shared_emb_functional.detach())
        loss_ortho_visual = orthogonal_loss(z_specific_visual, shared_emb_visual.detach())


        loss_dict = {
            "recon_functional": loss_recon_functional,
            "recon_visual": loss_recon_visual,
            "ortho_functional": loss_ortho_functional,
            "ortho_visual": loss_ortho_visual,
            "total": (loss_recon_functional + loss_recon_visual) * 0.5 + (loss_ortho_functional + loss_ortho_visual) * 1.0
        }

        return loss_dict, z_specific_functional, z_specific_visual

    @torch.no_grad()
    def get_specific_embeddings(self, functional_feats: torch.Tensor, visual_feats: torch.Tensor):

        self.eval()
        z_specific_functional, _, _ = self.functional_branch(
            functional_feats,
            use_mask=False,
            compute_recon=False,
        )
        z_specific_visual, _, _ = self.visual_branch(
            visual_feats,
            use_mask=False,
            compute_recon=False,
        )
        return z_specific_functional, z_specific_visual
