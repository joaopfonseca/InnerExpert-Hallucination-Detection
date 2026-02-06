import torch


def llm_description(model):
    """Generate a basic description of the LLM model.

    Args:
        model: The language model object.

    Returns:
        A string description of the model.
    """
    model_name = model.config._name_or_path
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads

    description = (
        f"Model Name: {model_name}\n"
        f"Number of Layers: {num_layers}\n"
        f"Hidden Size: {hidden_size}\n"
        f"Number of Attention Heads: {num_attention_heads}\n"
        f"Attention Head Size: {hidden_size // num_attention_heads}\n"
        f"Total Parameters: {model.num_parameters():,}\n"
    )
    return description


def get_attentions(outputs, layer_idx=None, head_idx=None):
    """Extract attention matrices from model outputs.

    Args:
        outputs: Model outputs containing attention matrices.
        layer_idx: Optional; specific layer index to extract attention from.
        head_idx: Optional; specific head index to extract attention from.
    Returns:
        List of attention matrices for each layer.
        (batch_size, sequence_length, sequence_length)
        (batch_size, num_layers, num_heads, sequence_length, sequence_length)
    """
    attentions = outputs["attentions"]
    if attentions is None:
        raise ValueError(
            "Attentions are not available in the model outputs. Ensure that "
            "'output_attentions=True' is set during the model forward pass."
        )
    if layer_idx is not None and head_idx is not None:
        return attentions[layer_idx][:, head_idx]
    else:
        return torch.swapaxes(torch.stack(attentions), 0, 1)


def get_hidden_states(outputs, layer_idx=None):
    """
    Extract hidden states from model outputs.

    Args:
        outputs: Model outputs containing hidden states.
        layer_idx: Optional; specific layer index to extract hidden states from.

    Returns:
        Array of hidden states with shape (initial embedding layer + num_layers,
        seq_length, hidden_size)
    """
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError(
            "Hidden states are not available in the model outputs. Ensure that"
            " 'output_hidden_states=True' is set during the model forward "
            "pass."
        )
    if layer_idx is not None:
        return hidden_states[layer_idx]
    else:
        hidden_states = torch.stack(
            [hidden_state.squeeze() for hidden_state in hidden_states]
        )
        hidden_states = (
            torch.swapaxes(hidden_states, 0, 1)
            if hidden_states.ndim == 4
            else hidden_states.reshape(1, *hidden_states.shape)
        )
        return hidden_states


def generate_params(
    inputs,
    tokenizer,
    output_attentions=True,
    output_hidden_states=True,
    output_scores=True,
    output_router_logits=True,
    **kwargs,
):
    return {
        **inputs,
        "attention_mask": inputs["attention_mask"],
        # "top_p":0.9,
        # "do_sample":True,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "output_scores": output_scores,
        "output_router_logits": output_router_logits,
        "use_cache": True,
        **kwargs,
    }


def format_outputs(outputs):
    pass
