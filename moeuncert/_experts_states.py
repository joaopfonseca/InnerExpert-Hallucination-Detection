import torch
from torch import nn
from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct", torch_dtype=torch.float16, device_map="auto")
# moe_layer = model.model.layers.block_sparse_moe  # adjust path to the MoE block you want

def modify_moe_block(moe_block):
    """
    Monkey-patch the forward method of the given MoE block to save intermediate
    expert hidden states.

    Note that the exact implementation of the forward method may depend on the
    specific architecture of the MoE block in your model. This is an attempt to
    provide a general structure, but you may need to adjust it based on how
    your model's MoE block is implemented.
    """

    class MoECustom(type(moe_block)):
        pass
    
    MoECustom.forward = forward_olmoe
    moe_block.__class__ = MoECustom
    return moe_block



def forward_olmoe(self, hidden_states, *args, **kwargs):
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
    experts_hidden = {}

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
        gate, up = nn.functional.linear(current_state, self.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = self.experts.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, self.experts.down_proj[expert_idx])

        # MODIFIED: Save the intermediate hidden states for this expert
        experts_hidden[int(expert_idx)] = current_hidden_states.detach().cpu()

        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )
    
    final_hidden_states = final_hidden_states.reshape(
        batch_size, sequence_length, hidden_dim
    )

    # MODIFIED: Save the experts' hidden states before the final combination step
    self.last_experts_hidden = experts_hidden

    return final_hidden_states
