"""
Model forward functions for MoE blocks.

These are used to replace the original forward methods of MoE blocks so that
per-expert intermediate hidden states are saved on `last_experts_hidden` for
downstream uncertainty metrics.

Currently supported MoE block classes:
  - OlmoeSparseMoeBlock (OLMoE-1B-7B-0924-Instruct)
  - Gemma4TextExperts  (Gemma 4 26B A4B IT)

To add a new model, write a new `forward_*` function below and register its
target class in `moeuncert.forwards._experts_states.MOE_FORWARD_REGISTRY`.
"""

import torch
import torch.nn.functional as F
from torch import nn


def forward_olmoe(self, hidden_states):
    """
    Forward pass for the OLMoE MoE block, modified to save per-expert hidden
    states before the routing-weighted combination step.

    Original structure:
        hidden_states -> gate (router) -> top_k_weights, selected_experts
        for expert in selected:
            out = expert(act_fn(gate_proj) * up_proj)  # via down_proj
            final += out * routing_weight
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    num_experts = self.experts.num_experts
    top_k = self.gate.top_k

    _, routing_weights, selected_experts = self.gate(hidden_states)
    routing_weights = routing_weights.to(hidden_states.dtype)

    experts_hidden = {
        "expert_idx": selected_experts.detach(),
        "expert_weights": routing_weights.detach(),
        "expert_hidden_states": torch.zeros(
            *routing_weights.shape, hidden_dim, dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).detach(),
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
        ] = current_hidden_states.detach()

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


def forward_gemma4(self, hidden_states, top_k_index, top_k_weights):
    """
    Forward pass for the Gemma 4 MoE experts module, modified to save
    per-expert hidden states before the routing-weighted combination step.

    Original structure (Gemma4TextExperts.forward):
        final = zeros_like(hidden_states)
        for expert in selected:
            out = expert(act_fn(gate_up_proj[expert][0]) * gate_up_proj[expert][1])
            out = down_proj[expert] @ out
            final += out * top_k_weights[token_idx, top_k_pos]
        return final

    Notes
    -----
    - Gemma 4's MoE runs *alongside* a dense shared MLP (Gemma4TextMLP) in the
      decoder layer. The shared MLP is NOT instrumented here; it is computed
      normally by the layer and added to this module's output. Only the
      routed experts are captured for uncertainty metrics.
    - 128 total experts, 8 active per token, 1 shared dense MLP (per the
      published Gemma 4 architecture).
    - The host layer (Gemma4TextDecoderLayer) flattens its hidden states from
      (B, S, H) to (B*S, H) before calling experts.forward. We recover the
      3D shape via `_current_batch_size` / `_current_seq_length`, which are
      stashed on this module by a pre-forward hook installed in
      `_experts_states._install_parent_shape_capture`.
    """
    num_experts = self.num_experts
    hidden_dim = self.hidden_dim
    num_tokens = top_k_index.shape[0]
    top_k = top_k_index.shape[-1]

    # Recover batch/seq from the parent layer's pre-hook (always 2D here).
    batch_size = getattr(self, "_current_batch_size", None)
    sequence_length = getattr(self, "_current_seq_length", None)
    if (
        batch_size is not None
        and sequence_length is not None
        and batch_size * sequence_length == num_tokens
    ):
        flat_hidden = hidden_states.reshape(num_tokens, hidden_dim) \
            if hidden_states.dim() == 3 else hidden_states
    else:
        # Fallback: try to infer from hidden_states if 3D.
        if hidden_states.dim() == 3:
            batch_size, sequence_length, _ = hidden_states.shape
            flat_hidden = hidden_states.reshape(num_tokens, hidden_dim)
        else:
            batch_size = None
            sequence_length = None
            flat_hidden = hidden_states

    experts_hidden = {
        "expert_idx": top_k_index.detach(),
        "expert_weights": top_k_weights.detach(),
        "expert_hidden_states": torch.zeros(
            num_tokens, top_k, hidden_dim, dtype=flat_hidden.dtype,
            device=flat_hidden.device,
        ).detach(),
    }

    final_hidden_states = torch.zeros_like(flat_hidden)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = flat_hidden[token_idx]

        gate, up = F.linear(
            current_state, self.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = F.linear(
            current_hidden_states, self.down_proj[expert_idx]
        )

        # Save pre-routing intermediate state for downstream metrics.
        experts_hidden["expert_hidden_states"][
            token_idx, top_k_pos
        ] = current_hidden_states.detach()

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    # Reshape to (B, S, top_k, ...) if we know the dimensions.
    if batch_size is not None and sequence_length is not None:
        self.last_experts_hidden = {
            "expert_idx": experts_hidden["expert_idx"].view(
                batch_size, sequence_length, top_k
            ),
            "expert_weights": experts_hidden["expert_weights"].view(
                batch_size, sequence_length, top_k
            ),
            "expert_hidden_states": experts_hidden["expert_hidden_states"].view(
                batch_size, sequence_length, top_k, hidden_dim
            ),
        }
    else:
        # Caller didn't provide batch/seq dimensions; keep flat layout.
        self.last_experts_hidden = {
            "expert_idx": experts_hidden["expert_idx"],
            "expert_weights": experts_hidden["expert_weights"],
            "expert_hidden_states": experts_hidden["expert_hidden_states"],
        }

    return final_hidden_states
