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
    Forward pass for the MoE block specifically for OLMoE, modified to save
    intermediate expert hidden states.

    Should replace the original forward method of the OlmoeSparseMoeBlock
    block.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    # MODIFIED: Save the intermediate hidden states for each expert before combining them
    experts_hidden = {
        "expert_idx": selected_experts.detach().cpu(),  # (num_tokens, top_k)
        "expert_weights": routing_weights.detach().cpu(),  # (num_tokens, top_k)
        "hidden_states": torch.zeros(
            *routing_weights.shape, hidden_dim, dtype=hidden_states.dtype
        ).detach().cpu(),  # (num_tokens, top_k, hidden_dim)
    }

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be selected
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, 
        num_classes=self.num_experts
    ).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) # MODIFIED

        # MODIFIED: Save the intermediate hidden states for this expert before weighting
        experts_hidden["hidden_states"][top_x, idx] = current_hidden_states.detach().cpu()

        # MODIFIED
        current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(final_hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    # MODIFIED: Save the experts' hidden states before the final combination step
    self.last_experts_hidden = experts_hidden

    return final_hidden_states, router_logits
