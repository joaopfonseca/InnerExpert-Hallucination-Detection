"""
At the moment this is being used to save the intermediate hidden states of the
experts only in the OLMoE model, but the logic can be adapted to other MoE
models as well.
"""

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from ._model_forwards import forward_olmoe


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

    Instance attributes (top_k, num_experts, experts, norm_topk_prob) are
    captured from the original object and stored as explicit instance attrs
    so that the monkey-patched forward can access them regardless of the
    transformers version.
    """
    # Capture needed attributes before the class swap destroys access via __class__
    for attr in ("top_k", "num_experts", "experts", "norm_topk_prob"):
        if not hasattr(moe_block, attr):
            try:
                setattr(moe_block, attr, getattr(moe_block.config, attr, None))
            except Exception:
                pass

    class MoEBlockCustom(type(moe_block)):
        pass

    MoEBlockCustom.forward = forward_olmoe
    moe_block.__class__ = MoEBlockCustom

    # accelerate replaces module.forward with a wrapper that calls _old_forward directly,
    # bypassing the class-level patch above. Patch _old_forward too when present.
    if hasattr(moe_block, "_old_forward"):
        import types

        moe_block._original_old_forward = moe_block._old_forward
        moe_block._old_forward = types.MethodType(forward_olmoe, moe_block)

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
    if hasattr(moe_block, "_original_old_forward"):
        moe_block._old_forward = moe_block._original_old_forward
        del moe_block._original_old_forward
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
