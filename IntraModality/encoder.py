import torch
import torch.nn as nn
import torchvision.models as models


class MLPEncoder(nn.Module):


    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageEncoder(nn.Module):


    def __init__(
        self,
        latent_dim: int = 128,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.output_dim = latent_dim


        backbone_fn = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }
        if backbone not in backbone_fn:
            raise ValueError(f"Available backbones: {list(backbone_fn.keys())}")

        weights = "IMAGENET1K_V1" if pretrained else None
        base = backbone_fn[backbone](weights=weights)


        backbone_dim = base.fc.in_features
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        feat = self.backbone(x)
        feat = feat.flatten(1)
        return self.projector(feat)
