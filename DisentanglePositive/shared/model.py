import torch
import torch.nn as nn
import torch.nn.functional as F
from .projector import ModalProjector
from .loss import CrossModalContrastiveLoss


class UrbanRegionAligner(nn.Module):
    def __init__(
        self,
        functional_dim=256,
        visual_dim=256,
        hidden_dim=512,
        out_dim=256,
        functional_input_parts=[128, 128],
        visual_input_parts=[128, 128],
        warmup_epochs=500,
        temperature=0.08,
        positive_percentile=85.0,
        positive_threshold_min=0.45,
        positive_threshold_max=0.85,
        beta=4.0,
    ):
        super().__init__()

        self.proj_functional = ModalProjector(
            input_dim=functional_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            sub_dims=functional_input_parts,
        )
        self.proj_visual = ModalProjector(
            input_dim=visual_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            sub_dims=visual_input_parts,
        )

        self.loss_fn = CrossModalContrastiveLoss(
            temperature=temperature,
            warmup_epochs=warmup_epochs,
            positive_percentile=positive_percentile,
            positive_threshold_min=positive_threshold_min,
            thresh_max=positive_threshold_max,
            beta=beta,
        )

    def step_epoch(self):
        self.loss_fn.step_epoch()


    def forward(
        self,
        functional_features,
        visual_features,
    ):
        z_functional_raw = self.proj_functional(functional_features)
        z_visual_raw = self.proj_visual(visual_features)
        z_functional = F.normalize(z_functional_raw, dim=-1)
        z_visual = F.normalize(z_visual_raw, dim=-1)
        loss = self.loss_fn(
            z_functional,
            z_visual,
        )
        return loss, z_functional_raw, z_visual_raw
