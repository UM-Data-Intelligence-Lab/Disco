import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):


    def __init__(self, temperature, warmup_epochs, positive_percentile,
                 positive_threshold_min, thresh_max, beta):
        super().__init__()
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs
        self.positive_percentile = positive_percentile
        self.positive_threshold_min = positive_threshold_min
        self.thresh_max = thresh_max
        self.beta = beta
        self.cur_epoch = 0

    def step_epoch(self):
        self.cur_epoch += 1

    @staticmethod
    def sim(z1, z2):
        return torch.mm(z1, z2.t())


    @torch.no_grad()
    def _mine_positive_weights(
        self,
        z_functional: torch.Tensor,
        z_visual: torch.Tensor,
    ) -> torch.Tensor:


        N = z_functional.size(0)
        device = z_functional.device

        z_mean = F.normalize((z_functional + z_visual) / 2.0, dim=-1)

        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        neg_mask = ~self_mask

        raw_mean = self.sim(z_mean, z_mean)

        q = self.positive_percentile / 100.0
        neg_vals = raw_mean[neg_mask]

        base_thresh = self.positive_threshold_min

        thresh = (
            float(
                torch.quantile(neg_vals, q)
            )
            if neg_vals.numel() > 0
            else base_thresh
        )

        thresh = max(min(thresh, self.thresh_max), self.positive_threshold_min)
        positive_mask = (raw_mean >= thresh) & (~self_mask)

        W = raw_mean.clamp(min=0.0)
        W[~positive_mask] = 0.0

        if self.beta != 1.0:
            W = W * self.beta

        return W


    def _semi_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        positive_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:


        device = z1.device
        N = z1.size(0)
        f = lambda x: torch.exp(x / self.temperature)

        B_mat = f(self.sim(z1, z2))
        R_mat = f(self.sim(z1, z1))

        self_mask = torch.eye(
            N,
            dtype=torch.bool,
            device=device,
        )
        neg_mask = ~self_mask

        denom = (
            (B_mat * neg_mask.float()).sum(1)
            + (R_mat * neg_mask.float()).sum(1)
        )

        pos_diag = B_mat.diagonal()

        diag_loss = -torch.log(
            pos_diag / (pos_diag + denom + 1e-8)
        )

        if positive_weights is None:
            return diag_loss.mean()


        full_denom = pos_diag + denom

        positive_loss_mat = -torch.log(
            B_mat / (full_denom.unsqueeze(1) + 1e-8)
        )

        weighted_sum = (
            positive_loss_mat * positive_weights
        ).sum(1)

        weight_count = positive_weights.sum(1)
        has_positive = weight_count > 0

        if has_positive.any():
            positive_loss = torch.where(
                has_positive,
                weighted_sum
                / weight_count.clamp(min=1e-8),
                diag_loss,
            )
            return positive_loss.mean()

        return diag_loss.mean()

    def _semi_loss_mean(
        self,
        z_mean: torch.Tensor,
        positive_weights: torch.Tensor,
    ) -> torch.Tensor:


        device = z_mean.device
        dtype = z_mean.dtype
        N = z_mean.size(0)

        positive_weights = positive_weights.to(
            device=device,
            dtype=dtype,
        )

        logits = self.sim(z_mean, z_mean) / self.temperature
        exp_logits = torch.exp(logits)

        self_mask = torch.eye(
            N,
            dtype=torch.bool,
            device=device,
        )

        pos_mask = (positive_weights > 0) & (~self_mask)
        neg_mask = (~self_mask) & (~pos_mask)


        neg_sum = (
            exp_logits * neg_mask.to(dtype)
        ).sum(dim=1)


        pair_denom = (
            exp_logits + neg_sum.unsqueeze(1)
        )

        pair_loss = -torch.log(
            exp_logits / (pair_denom + 1e-8)
        )


        positive_weights = (
            positive_weights * pos_mask.to(dtype)
        )

        weighted_positive_loss = (
            pair_loss * positive_weights
        ).sum(dim=1)

        positive_weight_sum = positive_weights.sum(dim=1)
        has_positive = positive_weight_sum > 0

        if not has_positive.any():


            return z_mean.sum() * 0.0

        per_anchor_loss = torch.zeros(
            N,
            device=device,
            dtype=dtype,
        )

        per_anchor_loss[has_positive] = (
            weighted_positive_loss[has_positive]
            / positive_weight_sum[has_positive].clamp(min=1e-8)
        )


        return per_anchor_loss[has_positive].mean()

    def forward(self, functional_embs, visual_embs):
        if functional_embs.size(0) < 2:
            return torch.tensor(0.0, device=functional_embs.device, requires_grad=True)
        z_functional = F.normalize(functional_embs, dim=-1)
        z_visual = F.normalize(visual_embs, dim=-1)
        if self.cur_epoch <= self.warmup_epochs:
            l1 = self._semi_loss(z_functional, z_visual)
            l2 = self._semi_loss(z_visual, z_functional)
            return (l1 + l2) * 0.5
        positive_weights = self._mine_positive_weights(z_functional, z_visual)
        z_mean = F.normalize((z_functional + z_visual) / 2.0, dim=-1)
        return self._semi_loss_mean(z_mean, positive_weights)
