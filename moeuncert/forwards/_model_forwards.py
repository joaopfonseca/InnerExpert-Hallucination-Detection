"""
Model forward functions for OLMoE. These are used to replace the original forward
methods of the OLMoE model and its MoE blocks to include logic for saving intermediate
expert hidden states.

Later on we can add forward passes to match other MoE models as well, but for now it's specifically
designed for OLMoE.
"""

import torch
import torch.nn.functional as F
from torch import nn


def forward_olmoe(self, hidden_states):
    """
    Forward pass for the MoE block, modified to save per-expert hidden states
    before the routing-weighted combination step.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    num_experts = self.experts.num_experts
    top_k = self.gate.top_k

    _, routing_weights, selected_experts = self.gate(hidden_states)
    routing_weights = routing_weights.to(hidden_states.dtype)

    experts_hidden = {
        "expert_idx": selected_experts.detach().cpu(),
        "expert_weights": routing_weights.detach().cpu(),
        "expert_hidden_states": torch.zeros(
            *routing_weights.shape, hidden_dim, dtype=hidden_states.dtype
        ).detach().cpu(),
    }

    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        gate, up = nn.functional.linear(
            current_state, self.experts.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        current_hidden_states = self.experts.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(
            current_hidden_states, self.experts.down_proj[expert_idx]
        )

        experts_hidden["expert_hidden_states"][
            token_idx, top_k_pos
        ] = current_hidden_states.detach().cpu()

        current_hidden_states = (
            current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )

    self.last_experts_hidden = {
        "expert_idx": experts_hidden["expert_idx"].view(batch_size, sequence_length, -1),
        "expert_weights": experts_hidden["expert_weights"].view(batch_size, sequence_length, -1),
        "expert_hidden_states": experts_hidden["expert_hidden_states"].view(
            batch_size, sequence_length, top_k, hidden_dim
        ),
    }

    return final_hidden_states
