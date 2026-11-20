import torch
import torch.nn as nn
import torch.nn.functional as F


class DownstreamMoEModel(nn.Module):

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 1,
        hidden_dim: int = 128,
        moe_output_dim: int = 384,
        gate_hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        if moe_output_dim <= 0 or moe_output_dim % 3 != 0:
            raise ValueError(
                "moe_output_dim must be positive and divisible by 3. "
                f"Received {moe_output_dim}"
            )

        expert_dim = moe_output_dim // 3

        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.moe_output_dim = moe_output_dim
        self.gate_hidden_dim = gate_hidden_dim
        self.dropout = dropout

        self.gate_norm_shared = nn.LayerNorm(emb_dim)
        self.gate_norm_functional = nn.LayerNorm(emb_dim)
        self.gate_norm_visual = nn.LayerNorm(emb_dim)

        self.gating_network = nn.Sequential(
            nn.Linear(emb_dim * 3, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 3),
        )

        with torch.no_grad():
            self.gating_network[-1].bias.data[0] = 1.0
            self.gating_network[-1].bias.data[1] = 0.0
            self.gating_network[-1].bias.data[2] = 0.0

        self.expert_shared = nn.Linear(emb_dim, expert_dim)
        self.expert_functional = nn.Linear(emb_dim, expert_dim)
        self.expert_visual = nn.Linear(emb_dim, expert_dim)

        self.expert_norm_shared = nn.LayerNorm(expert_dim)
        self.expert_norm_functional = nn.LayerNorm(expert_dim)
        self.expert_norm_visual = nn.LayerNorm(expert_dim)

        self.predict_head = nn.Sequential(
            nn.Linear(moe_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        z_shared: torch.Tensor,
        z_functional: torch.Tensor,
        z_visual: torch.Tensor,
    ):
        z_shared_gate = self.gate_norm_shared(z_shared)
        z_functional_gate = self.gate_norm_functional(z_functional)
        z_visual_gate = self.gate_norm_visual(z_visual)

        gate_input = torch.cat([z_shared_gate, z_functional_gate, z_visual_gate], dim=-1)
        gate_logits = self.gating_network(gate_input)
        gates = torch.sigmoid(gate_logits)

        e_shared = F.gelu(self.expert_norm_shared(self.expert_shared(z_shared)))
        e_functional = F.gelu(self.expert_norm_functional(self.expert_functional(z_functional)))
        e_visual = F.gelu(self.expert_norm_visual(self.expert_visual(z_visual)))

        experts = torch.stack([e_shared, e_functional, e_visual], dim=1)
        weighted_experts = gates.unsqueeze(-1) * experts
        z_fused = weighted_experts.flatten(start_dim=1)

        preds = self.predict_head(z_fused)
        return preds, gates
