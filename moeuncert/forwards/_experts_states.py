"""
At the moment this is being used to save the intermediate hidden states of the
experts only in the OLMoE model, but the logic can be adapted to other MoE
models as well.
"""

import types

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from ._model_forwards import forward_olmoe


def modify_model(model):
    """
    Modify the given model to save intermediate expert hidden states in its MoE blocks.
    """
    modify_model_forward_method(model)

    for module in model.modules():
        if isinstance(module, OlmoeSparseMoeBlock):
            modify_moe_block(module)
    return model


def modify_model_forward_method(model):
    """
    Wrap the model's forward method to collect expert hidden states after each call.

    Stored as a plain function (not types.MethodType) so nn.Module.__call__ invokes
    it without double-binding. The original forward (accelerate hook) is captured
    as a bound method in the closure.
    """
    model._original_forward_custom = model.forward
    orig_forward = model._original_forward_custom

    def new_forward(*args, **kwargs):
        outputs = orig_forward(*args, **kwargs)
        if hasattr(model, 'model'):
            outputs["experts_hidden"] = [
                layer.mlp.last_experts_hidden for layer in model.model.layers
            ]
        return outputs

    model.forward = new_forward


def modify_moe_block(moe_block):
    """
    Replace the MoE block's forward method to save intermediate expert hidden states.

    Uses types.MethodType to bind forward_olmoe directly to the original instance,
    avoiding __class__ replacement which breaks attribute access in newer transformers.
    """
    moe_block._original_forward = moe_block.forward
    moe_block.forward = types.MethodType(forward_olmoe, moe_block)

    if hasattr(moe_block, "_old_forward"):
        moe_block._original_old_forward = moe_block._old_forward
        moe_block._old_forward = types.MethodType(forward_olmoe, moe_block)

    return moe_block


def reset_model(model):
    """
    Reset the modifications made to the model by `modify_model`.
    """
    if hasattr(model, "_original_forward_custom"):
        model.forward = model._original_forward_custom
        del model._original_forward_custom

    for module in model.modules():
        if isinstance(module, OlmoeSparseMoeBlock):
            reset_moe_block(module)

    return model


def reset_moe_block(moe_block):
    """
    Restore the original forward method of the MoE block.
    """
    if hasattr(moe_block, "_original_forward"):
        moe_block.forward = moe_block._original_forward
        del moe_block._original_forward
    if hasattr(moe_block, "_original_old_forward"):
        moe_block._old_forward = moe_block._original_old_forward
        del moe_block._original_old_forward
