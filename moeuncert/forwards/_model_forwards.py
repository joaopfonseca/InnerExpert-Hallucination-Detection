import torch
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
    _, top_k_weights, top_k_index = self.gate(hidden_states)

    # This replaces the self.experts call in the original forward
    # final_hidden_states = self.experts(
    #         hidden_states,
    #         top_k_index,
    #         top_k_weights
    # ).reshape(batch_size, sequence_length, hidden_dim)

    # MODIFIED: Save the intermediate hidden states for each expert before combining them
    experts_hidden = {
        "expert_idx": top_k_index.detach().cpu(),  # (num_tokens, indices top_k)
        "expert_weights": top_k_weights.detach().cpu(),  # (num_tokens, weights top_k)
        "hidden_states": torch.zeros(
            *top_k_weights.shape, hidden_dim, dtype=hidden_states.dtype
        )
        .detach()
        .cpu(),  # (num_tokens, top_k, hidden_dim)
    }
    # print(
    #     "\n==============================\n",
    #     experts_hidden["expert_idx"],
    #     "\n==============================",
    # )
    # print(hidden_states.shape, top_k_weights.shape, top_k_index.shape)

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.experts.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.experts.num_experts:
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

        # MODIFIED: Save the intermediate hidden states for this expert
        # print(current_hidden_states.shape, token_idx, top_k_pos, expert_idx)
        experts_hidden["hidden_states"][
            token_idx, top_k_pos
        ] = current_hidden_states.detach().cpu()

        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )

    # MODIFIED: Save the experts' hidden states before the final combination step
    self.last_experts_hidden = experts_hidden

    return final_hidden_states
