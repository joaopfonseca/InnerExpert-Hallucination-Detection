"""
At the moment this is being used to save the intermediate hidden states of the
experts in MoE blocks, with per-architecture dispatch via a class registry.

Currently registered MoE block classes:
  - OlmoeSparseMoeBlock   (OLMoE-1B-7B-0924-Instruct)
  - Gemma4TextExperts     (Gemma 4 26B A4B IT)

Adding a new model
------------------
1. Implement a new `forward_<model>` function in `_model_forwards.py`. The
   forward must set `self.last_experts_hidden` to a dict with keys
   `expert_idx`, `expert_weights`, `expert_hidden_states` matching the layout
   used by `forward_olmoe` (see `moeuncert/metrics/_metrics.py` for shapes).
2. Import the MoE block class below and add an entry to
   `MOE_FORWARD_REGISTRY`. That's it — `_llm_monitor.py` and `modify_model`
   will pick it up automatically.

If the new MoE block's host layer flattens the hidden states before calling
the experts (e.g., Gemma 4's `Gemma4TextDecoderLayer` flattens
`(B, S, H) -> (B*S, H)` before `self.experts(...)`), `modify_model` will
automatically register a pre-forward hook on the host layer that stashes
`(_current_batch_size, _current_seq_length)` on the experts module. The
forward function can then read these attributes to produce output tensors with
the correct 4D shape `(B, S, top_k, hidden)`.
"""

import types

import torch

from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

from ._model_forwards import forward_olmoe, forward_gemma4


# Optional: Gemma 4 is only available in newer transformers. Import
# defensively so the package still loads on older transformers versions
# (with OLMoE-only support).
try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    _HAS_GEMMA4 = True
except ImportError:
    Gemma4TextExperts = None
    _HAS_GEMMA4 = False


# Registry: MoE block class -> forward function that sets
# `self.last_experts_hidden` before returning the routed expert output.
MOE_FORWARD_REGISTRY = {OlmoeSparseMoeBlock: forward_olmoe}
if _HAS_GEMMA4:
    MOE_FORWARD_REGISTRY[Gemma4TextExperts] = forward_gemma4

# Tuple of classes whose forward methods we patch.
MOE_BLOCK_CLASSES = tuple(MOE_FORWARD_REGISTRY.keys())


def is_moe_block(module) -> bool:
    """Return True if `module` is an instance of a registered MoE block class."""
    return type(module) in MOE_FORWARD_REGISTRY


def modify_model(model):
    """
    Modify the given model to save intermediate expert hidden states in its
    MoE blocks. Dispatches to the correct forward based on each block's
    concrete class.
    """
    modify_model_forward_method(model)

    # Walk the model with parent context. For each MoE block, find the host
    # layer and register a pre-forward hook that stashes the layer's input
    # shape on the MoE block. This lets the patched MoE forward produce
    # output tensors with the correct 4D batch/seq layout, even if the host
    # layer flattens its hidden states before calling the experts.
    for parent, child in _walk_with_parent(model):
        if parent is not None and is_moe_block(child):
            _install_parent_shape_capture(parent, child)

    for module in model.modules():
        if is_moe_block(module):
            modify_moe_block(module)
    return model


def _walk_with_parent(model):
    """Yield every (parent_module, child_module) pair reachable from model.

    `child` is a direct child submodule of `parent`. The root `model` is
    yielded with a None parent.
    """
    stack = [(None, model)]
    while stack:
        parent, module = stack.pop()
        yield parent, module
        for child in module.children():
            stack.append((module, child))


def _install_parent_shape_capture(parent_layer, moe_block):
    """
    Register a pre-forward hook on `parent_layer` that captures its input
    shape and stashes `(_current_batch_size, _current_seq_length)` on
    `moe_block`. The MoE block's patched forward can then read these
    attributes to produce output tensors with the correct 4D layout.

    Only install the hook for blocks whose patched forward requires
    batch/sequence recovery. Currently that's `Gemma4TextExperts` whose
    host layer (`Gemma4TextDecoderLayer`) flattens hidden states from
    `(B, S, H)` to `(B*S, H)` before calling `self.experts(...)`.
    `OlmoeSparseMoeBlock` already gets 3D input from its host layer, so no
    hook is needed.
    """
    if not _block_needs_shape_capture(moe_block):
        return

    # Idempotency: don't double-install.
    if getattr(moe_block, "_shape_capture_hook", None) is not None:
        return

    def pre_hook(module, args, kwargs):
        # Gemma4TextDecoderLayer.forward signature:
        #   (hidden_states, per_layer_input=None, shared_kv_states=None,
        #    position_embeddings=None, attention_mask=None, position_ids=None,
        #    past_key_values=None, **kwargs)
        # The first positional arg is the 3D hidden_states.
        hidden_states = None
        if args:
            hidden_states = args[0]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3:
            moe_block._current_batch_size = hidden_states.shape[0]
            moe_block._current_seq_length = hidden_states.shape[1]
        # else: leave prior values (shouldn't happen in normal generation).

    handle = parent_layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    moe_block._shape_capture_hook = handle


# Set of block classes whose patched forward needs batch/sequence recovery.
# Add a class to this set when its host layer flattens hidden states before
# calling the block.
_SHAPE_CAPTURE_REQUIRED = set()
if _HAS_GEMMA4:
    _SHAPE_CAPTURE_REQUIRED.add(Gemma4TextExperts)


def _block_needs_shape_capture(moe_block) -> bool:
    return type(moe_block) in _SHAPE_CAPTURE_REQUIRED


def _uninstall_parent_shape_capture(moe_block):
    handle = getattr(moe_block, "_shape_capture_hook", None)
    if handle is not None:
        handle.remove()
        delattr(moe_block, "_shape_capture_hook")


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
        # Collect last_experts_hidden from every MoE block in the model.
        # Works for both OLMoE (layers[i].mlp) and Gemma 4
        # (layers[i].experts) — both expose `last_experts_hidden` after the
        # block-level forward has run.
        experts_hidden_list = []
        for module in model.modules():
            if is_moe_block(module) and hasattr(module, "last_experts_hidden"):
                experts_hidden_list.append(module.last_experts_hidden)
        if experts_hidden_list:
            # Only attach if `outputs` is dict-like (e.g., a ModelOutput
            # dataclass or a plain dict). The original code already handled
            # plain dicts; this guard also covers dataclass outputs.
            if isinstance(outputs, dict) or hasattr(outputs, "__setitem__"):
                try:
                    outputs["experts_hidden"] = experts_hidden_list
                except (TypeError, KeyError):
                    pass
        return outputs

    model.forward = new_forward


def modify_moe_block(moe_block):
    """
    Replace the MoE block's forward method with the architecture-specific
    forward that saves intermediate expert hidden states.

    Uses types.MethodType to bind the forward directly to the original
    instance, avoiding __class__ replacement which breaks attribute access in
    newer transformers.
    """
    forward_fn = MOE_FORWARD_REGISTRY.get(type(moe_block))
    if forward_fn is None:
        raise ValueError(
            f"No forward registered for MoE block class {type(moe_block).__name__}. "
            f"Add it to MOE_FORWARD_REGISTRY in moeuncert.forwards._experts_states."
        )

    if not hasattr(moe_block, "_original_forward"):
        moe_block._original_forward = moe_block.forward
    moe_block.forward = types.MethodType(forward_fn, moe_block)

    if hasattr(moe_block, "_old_forward") and not hasattr(moe_block, "_original_old_forward"):
        moe_block._original_old_forward = moe_block._old_forward
        moe_block._old_forward = types.MethodType(forward_fn, moe_block)

    return moe_block


def reset_model(model):
    """
    Reset the modifications made to the model by `modify_model`.
    """
    if hasattr(model, "_original_forward_custom"):
        model.forward = model._original_forward_custom
        del model._original_forward_custom

    for module in model.modules():
        if is_moe_block(module):
            reset_moe_block(module)
            _uninstall_parent_shape_capture(module)

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
    # Clear the cached outputs from the most recent forward call.
    if hasattr(moe_block, "last_experts_hidden"):
        del moe_block.last_experts_hidden
    # Clear the captured shape stashed by the parent-layer pre-hook.
    for attr in ("_current_batch_size", "_current_seq_length"):
        if hasattr(moe_block, attr):
            delattr(moe_block, attr)
