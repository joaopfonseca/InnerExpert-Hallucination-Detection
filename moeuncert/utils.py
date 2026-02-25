import torch
import torch.nn.functional as F


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


def move_to_device(data, device="cpu"):
    """
    Move tensors to CPU. If the input is a list-like or dict-like, it recursively
    moves each item to CPU.
    """
    if type(data) in [int, float, str]:
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif (
        hasattr(data, "__len__")
        and hasattr(data, "__getitem__")
        and not hasattr(data, "items")
    ):
        return type(data)([move_to_device(item, device=device) for item in data])
    elif hasattr(data, "items"):
        return {
            move_to_device(key, device=device): move_to_device(value, device=device)
            for key, value in data.items()
        }
    else:
        raise ValueError(
            f"Unsupported data type: {type(data)}. Expected torch.Tensor, "
            "list-like, or dict-like."
        )

    return data


def standardize_outputs(outputs, device=None):
    """
    Standardizes the outputs of a MoE model to a consistent format.

    Output shapes:
    - sequences: (batch_size, seq_len)
    - hidden_states: (batch_size, seq_len, n_layers, hidden_size)
    - attentions: (batch_size, n_layers, num_heads, seq_len, seq_len)
    - scores: (batch_size, gen_seq_len, vocab_size)

    """
    processed_outputs = {}
    processed_outputs["sequences"] = outputs["sequences"]  # (batch_size, seq_len)

    if "hidden_states" in outputs:
        # Original shape: (gen_seq_len, n_layers, batch_size, 1, hidden_size)
        # outputs["hidden_states"]

        # (n_layers, batch_size, seq_len, hidden_size)
        hidden_states = torch.concat(
            [torch.stack(hs) for hs in outputs["hidden_states"]], dim=-2
        )
        # (batch_size, seq_len, n_layers, hidden_size)
        processed_outputs["hidden_states"] = hidden_states.permute(1, 2, 0, 3)

    # attentions
    if "attentions" in outputs:
        # Original shape (que): (gen_seq_len, n_layers, batch_size, num_heads, que_len, que_len)
        # Original shape (gen): (gen_seq_len, n_layers, batch_size, num_heads, 1, seq_len_curr)
        max_len = outputs["sequences"].shape[1] - 1

        # Pad the attention matrices to ensure they have the same shape:
        # (..., batch_size, num_heads, -1, max_len)
        attentions = [
            [
                F.pad(attn, (0, max_len - attn.shape[-1]), value=0)
                for attn in attn_layers
            ]
            for attn_layers in outputs["attentions"]
        ]

        # Concat the attentions across layers and heads
        # (n_layers, batch_size, num_heads, seq_len, seq_len)
        attentions = torch.concat([torch.stack(attn) for attn in attentions], dim=-2)

        # Permute to (batch_size, n_layers, num_heads, seq_len, seq_len)
        processed_outputs["attentions"] = attentions.permute(1, 0, 2, 3, 4)

    if "scores" in outputs:
        # Original shape: (gen_seq_len, batch_size, vocab_size)
        # outputs["scores"]
        # Permute to: (batch_size, gen_seq_len, vocab_size)
        processed_outputs["scores"] = torch.stack(outputs["scores"]).permute(1, 0, 2)

    if "router_logits" in outputs:
        pass

    if "experts_hidden" in outputs:
        pass

    if device is not None:
        processed_outputs = move_to_device(processed_outputs, device=device)

    return processed_outputs
