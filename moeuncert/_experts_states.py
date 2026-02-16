"""
At the moment this is being used to save the intermediate hidden states of the
experts only in the OLMoE model, but the logic can be adapted to other MoE
models as well.
"""

import torch
from torch import nn

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock


def modify_model(model):
    """
    Modify the given model to save intermediate expert hidden states in its MoE blocks.

    This function assumes that the model has a specific structure where MoE blocks can be
    identified and modified. You may need to adjust the logic for identifying and
    modifying the MoE blocks based on the actual architecture of your model.
    """
    modify_model_forward_method(model)

    for module in model.modules():
        if isinstance(module, OlmoeSparseMoeBlock):
            modify_moe_block(module)
    return model


def modify_model_forward_method(model):
    """
    Modify the forward method of the model to include logic for saving intermediate
    expert hidden states. This is a general approach that can be applied to
    any MoE model, but the forward method for the MoE layers always needs to be
    adapted based on the
    specific architecture of your model and how the MoE blocks are integrated.
    """

    class MoECustomForCausalLM(type(model)):
        pass

    MoECustomForCausalLM.original_forward = model.forward
    MoECustomForCausalLM.forward = main_forward_moe
    model.__class__ = MoECustomForCausalLM


def modify_moe_block(moe_block):
    """
    Monkey-patch the forward method of the given MoE block to save intermediate
    expert hidden states.

    Note that the exact implementation of the forward method may depend on the
    specific architecture of the MoE block in your model. This is an attempt to
    provide a general structure, but you may need to adjust it based on how
    your model's MoE block is implemented.
    """

    class MoEBlockCustom(type(moe_block)):
        pass

    MoEBlockCustom.forward = forward_olmoe
    moe_block.__class__ = MoEBlockCustom
    return moe_block


def reset_model(model):
    """
    Reset the modifications made to the model by `modify_model`, restoring the
    original forward methods of the MoE blocks.
    """

    model.__class__ = type(model).__bases__[0]  # Reset to original class

    for module in model.modules():
        if isinstance(module, OlmoeSparseMoeBlock):
            reset_moe_block(module)

    return model


def reset_moe_block(moe_block):
    """
    Reset the forward method of the given MoE block to its original
    implementation. This is useful if you want to revert the changes made by
    `modify_moe_block`.
    """
    moe_block.__class__ = type(moe_block).__bases__[0]  # Reset to original class


def main_forward_moe(self, *args, **kwargs):
    """
    Main forward method for the model that includes logic to save intermediate
    expert hidden states.
    """
    outputs = self.original_forward(*args, **kwargs)
    experts_hidden = [layer.mlp.last_experts_hidden for layer in self.model.layers]
    outputs["experts_hidden"] = experts_hidden
    return outputs


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
