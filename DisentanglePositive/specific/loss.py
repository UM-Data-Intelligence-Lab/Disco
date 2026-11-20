import torch
import torch.nn.functional as F

def orthogonal_loss(z_specific: torch.Tensor, z_shared: torch.Tensor) -> torch.Tensor:


    z_p_norm = F.normalize(z_specific, dim=-1)
    z_s_norm = F.normalize(z_shared, dim=-1)


    cosine_sim = (z_p_norm * z_s_norm).sum(dim=-1)
    return (cosine_sim ** 2).mean()

def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:


    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


    pred_masked = pred[mask]
    target_masked = target[mask]

    return F.mse_loss(pred_masked, target_masked)
