import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .shared.model import UrbanRegionAligner
from .specific.model import ModalitySpecificExtractor
from .reconstruction import GlobalReconstructor


@dataclass
class TrainConfig:
    seed: int
    num_epochs: int
    warmup_epochs: int
    mining_burnin: int
    positive_percentile: float
    positive_threshold_min: float
    positive_threshold_max: float
    mask_ratio: float
    batch_size: int
    lr: float
    weight_decay: float
    temperature: float
    edge_weight_scale: float
    functional_dim: int
    visual_dim: int
    functional_input_parts: tuple
    visual_input_parts: tuple
    hidden_dim: int
    out_dim: int
    specific_dim: int
    lambda_contrast: float
    lambda_ortho: float
    lambda_mask: float
    lambda_global: float
    device: str


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_grad_norm(model: nn.Module) -> float:

    total_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().norm(2).item()
        total_sq += grad_norm * grad_norm
    return total_sq**0.5


class JointDisentangledModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        self.shared_branch = UrbanRegionAligner(
            functional_dim=cfg.functional_dim,
            visual_dim=cfg.visual_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
            functional_input_parts=list(cfg.functional_input_parts),
            visual_input_parts=list(cfg.visual_input_parts),
            warmup_epochs=cfg.warmup_epochs,
            temperature=cfg.temperature,
            positive_percentile=cfg.positive_percentile,
            positive_threshold_min=cfg.positive_threshold_min,
            positive_threshold_max=cfg.positive_threshold_max,
            beta=cfg.edge_weight_scale,
        )


        self.specific_branch = ModalitySpecificExtractor(
            functional_dim=cfg.functional_dim,
            visual_dim=cfg.visual_dim,
            specific_dim=cfg.specific_dim,
            mask_ratio=cfg.mask_ratio,
        )

        self.global_recon_functional = GlobalReconstructor(
            shared_dim=cfg.out_dim,
            specific_dim=cfg.specific_dim,
            target_dim=cfg.functional_dim,
            hidden_dim=cfg.hidden_dim,
        )
        self.global_recon_visual = GlobalReconstructor(
            shared_dim=cfg.out_dim,
            specific_dim=cfg.specific_dim,
            target_dim=cfg.visual_dim,
            hidden_dim=cfg.hidden_dim,
        )


    def step_epoch(self):
        self.shared_branch.step_epoch()

    def forward(self, functional_orig, visual_orig):


        loss_contrast, z_shared_functional_raw, z_shared_visual_raw = self.shared_branch(
            functional_orig,
            visual_orig,
        )


        loss_fn = self.shared_branch.loss_fn
        in_positive_mining = loss_fn.cur_epoch > loss_fn.warmup_epochs


        if in_positive_mining:
            total_loss = (
                self.cfg.lambda_contrast * loss_contrast
            )

            zero = loss_contrast.detach().new_zeros(())

            loss_tracker = {
                "contrast": loss_contrast.item(),
                "ortho": zero.item(),
                "mask_raw": zero.item(),
                "mask_used": zero.item(),
                "global": zero.item(),
                "total": total_loss.item(),
            }

            return total_loss, loss_tracker


        z_shared = (z_shared_functional_raw + z_shared_visual_raw) / 2.0


        loss_specific_dict, z_functional, z_visual = self.specific_branch(
            functional_orig,
            visual_orig,
            z_shared_functional_raw.detach(),
            z_shared_visual_raw.detach(),
        )


        global_recon_functional = self.global_recon_functional(
            z_shared.detach(),
            z_functional,
        )
        global_recon_visual = self.global_recon_visual(
            z_shared.detach(),
            z_visual,
        )
        loss_global = (
            F.mse_loss(global_recon_functional, functional_orig)
            + F.mse_loss(global_recon_visual, visual_orig)
        )

        loss_ortho_total = (
            loss_specific_dict["ortho_functional"] + loss_specific_dict["ortho_visual"]
        )
        loss_mask_raw = (
            loss_specific_dict["recon_functional"] + loss_specific_dict["recon_visual"]
        )


        loss_mask_used = self.cfg.lambda_mask * loss_mask_raw

        total_loss = (
            self.cfg.lambda_contrast * loss_contrast
            + self.cfg.lambda_ortho * loss_ortho_total
            + loss_mask_used
            + self.cfg.lambda_global * loss_global
        )

        loss_tracker = {
            "contrast": loss_contrast.item(),
            "ortho": loss_ortho_total.item(),


            "mask_raw": loss_mask_raw.item(),
            "mask_used": loss_mask_used.item(),
            "global": loss_global.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_tracker

    @torch.no_grad()
    def get_all_embeddings(self, functional_orig, visual_orig):
        self.eval()
        z_shared_functional = self.shared_branch.proj_functional(functional_orig)
        z_shared_visual = self.shared_branch.proj_visual(visual_orig)
        z_shared = (z_shared_functional + z_shared_visual) / 2.0
        z_functional, z_visual = self.specific_branch.get_specific_embeddings(
            functional_orig,
            visual_orig,
        )
        return z_shared, z_functional, z_visual


def train_model(data: dict, cfg: TrainConfig, save_dir):
    city_save_dir = save_dir
    functional_tensor = torch.tensor(data["functional"])
    visual_tensor = torch.tensor(data["visual"])

    dataset = TensorDataset(
        functional_tensor,
        visual_tensor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        drop_last=True,
    )

    model = JointDisentangledModel(cfg).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
    )

    os.makedirs(city_save_dir, exist_ok=True)
    best_model_path = os.path.join(city_save_dir, "joint_model.pt")

    best_loss = float("inf")
    best_epoch = 0

    selection_start_epoch = (
        cfg.warmup_epochs + cfg.mining_burnin + 1
    )

    tracked_loss_names = [
        "contrast",
        "ortho",
        "mask_raw",
        "mask_used",
        "global",
        "total",
    ]

    for epoch in range(cfg.num_epochs):
        model.train()
        acc_losses = {name: 0.0 for name in tracked_loss_names}
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        epoch_lr = optimizer.param_groups[0]["lr"]

        for b_functional, b_visual in dataloader:
            b_functional = b_functional.to(cfg.device)
            b_visual = b_visual.to(cfg.device)

            optimizer.zero_grad()
            total_loss, loss_dict = model(
                b_functional,
                b_visual,
            )
            total_loss.backward()

            grad_norm = compute_grad_norm(model)
            grad_norm_sum += grad_norm
            grad_norm_max = max(grad_norm_max, grad_norm)

            optimizer.step()

            for name in tracked_loss_names:
                acc_losses[name] += loss_dict[name]

        n_batches = len(dataloader)
        avg_losses = {
            name: acc_losses[name] / n_batches
            for name in tracked_loss_names
        }
        avg_grad_norm = grad_norm_sum / n_batches

        epoch_num = epoch + 1
        eligible_for_best = epoch_num >= selection_start_epoch
        is_best = (
            eligible_for_best
            and avg_losses["total"] < best_loss
        )

        if is_best:
            best_loss = avg_losses["total"]
            best_epoch = epoch_num
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()
        model.step_epoch()

        if epoch_num % 10 == 0 or epoch == 0:
            print_phase = (
                f"Standard Warmup {epoch_num}/{cfg.warmup_epochs + 1}"
                if epoch <= cfg.warmup_epochs
                else "Positive Mining"
            )

            best_mark = " (best)" if is_best else ""
            mask_status = (
                f"raw={avg_losses['mask_raw']:.4f}, "
                f"used={avg_losses['mask_used']:.4f}"
            )

            print(
                f"\n   Epoch {epoch_num:3d}/{cfg.num_epochs} "
                f"| LR: {epoch_lr:.6f}"
            )
            print(
                f"     [Total]          "
                f"{avg_losses['total']:.4f}{best_mark}"
            )
            print(
                f"     [Contrast {print_phase}] "
                f"{avg_losses['contrast']:.4f}"
            )
            print(f"     [Ortho]          {avg_losses['ortho']:.4f}")
            print(f"     [Mask]           {mask_status}")
            print(f"     [Global Recon]   {avg_losses['global']:.4f}")
            print(
                f"     [Grad Norm]      mean={avg_grad_norm:.6f}, "
                f"max={grad_norm_max:.6f}"
            )

            if eligible_for_best and best_epoch > 0:
                print(
                    f"     [Best]           epoch={best_epoch}, "
                    f"loss={best_loss:.6f}"
                )

    best_state = torch.load(best_model_path, map_location=cfg.device)
    model.load_state_dict(best_state)

    print(f"Best epoch: {best_epoch}, loss: {best_loss:.6f}")

    z_shared, z_functional, z_visual = model.get_all_embeddings(
        functional_tensor.to(cfg.device),
        visual_tensor.to(cfg.device),
    )

    return (
        model,
        z_shared.cpu().numpy(),
        z_functional.cpu().numpy(),
        z_visual.cpu().numpy(),
    )

def fit(functional, visual, region_ids, config, device, output):
    cfg = TrainConfig(**config, device=device)
    set_seed(cfg.seed)
    data = {"functional": functional, "visual": visual}
    _, shared, functional, visual = train_model(data, cfg, output)
    for name, value in (("shared", shared), ("functional", functional), ("visual", visual)):
        np.save(output / f"{name}_embeddings.npy", value)
    np.save(output / "h3_order.npy", np.asarray(region_ids))
    return shared, functional, visual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "model.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads(args.config.read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    functional = np.load(args.data_dir / "functional_input.npy", allow_pickle=False)
    visual = np.load(args.data_dir / "visual_input.npy", allow_pickle=False)
    region_ids = np.load(args.data_dir / "h3_order.npy", allow_pickle=False).tolist()
    fit(functional, visual, region_ids, config, device, args.output_dir)


if __name__ == "__main__":
    main()
