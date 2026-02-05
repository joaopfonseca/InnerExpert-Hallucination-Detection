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
        hidden_states = torch.stack([hidden_state.squeeze() for hidden_state in hidden_states])
        hidden_states = (
            torch.swapaxes(hidden_states, 0, 1) 
            if hidden_states.ndim == 4 else hidden_states.reshape(1, *hidden_states.shape)
        )
        return hidden_states


def hidden_score(hidden_states):
    """
    Compute hidden state scores based on the LLM-Check method.

    Args:
        hidden_states: Tensor of hidden states with shape
        (num_layers + 1, seq_length, hidden_size)

    Returns:
        Tensor of hidden state scores with shape (num_layers + 1, seq_length)
    """

    hidden_states_transposed = torch.transpose(hidden_states, dim0=-2, dim1=-1)  # (batch_size, num_layers, hidden_size, seq_length)
    cov_matrices = hidden_states @ hidden_states_transposed  # (batch_size, num_layers, seq_length, seq_length)
    singular_values = torch.linalg.svd(cov_matrices).S  # (batch_size, num_layers, seq_length)
    score = 2*torch.cumsum(torch.log(singular_values), dim=-1)
    score /= (torch.arange(score.shape[-1])+1)

    return score


def attention_score(attentions):
    """
    Compute attention scores based on the LLM-Check method.

    Args:
        attentions: Tensor of attention matrices with shape
        (num_layers, num_heads, seq_length, seq_length)

    Returns:
        Tensor of attention scores with shape (num_layers, seq_length)
    """
    score = torch.cumsum(torch.log(attentions.diagonal(dim1=-2, dim2=-1)), dim=-1)
    return score


def generate_params(
    inputs, 
    tokenizer, 
    output_attentions=True,
    output_hidden_states=True,
    output_scores=True,
    output_router_logits=True,
    **kwargs
):
    return {
        **inputs,
        "attention_mask":inputs["attention_mask"],
        # "top_p":0.9,
        # "do_sample":True,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_attentions": output_attentions,
        "output_hidden_states": output_hidden_states,
        "output_scores": output_scores,
        "output_router_logits": output_router_logits,
        "use_cache": True,
        **kwargs
    }

def _unpack_input_outputs(outputs):
    """Get the logits, hidden states, and attentions for the last token in the input sequence."""
    outputs_sliced = {}
    if "logits" in outputs:
        outputs_sliced["logits"] = [outputs["logits"][0][:, -1]]
    if "hidden_states" in outputs:
        outputs_sliced["hidden_states"] = [[layer[:, -1, :] for layer in outputs["hidden_states"][0]]]
    if "attentions" in outputs:
        outputs_sliced["attentions"] = outputs["attentions"]
    if "router_logits" in outputs:
        outputs_sliced["router_logits"] = [[layer[-1] for layer in outputs["router_logits"][0]]]
    return {**outputs_sliced, **{k: v for k, v in outputs.items() if k not in outputs_sliced}}


def reconstruct_model_output(model, input_ids, output_ids, **model_kwargs):
    """
    Reconstruct the model outputs exactly as they occurred
    during autoregressive generation.
    
    Args:
        model: The language model
        input_ids: The input prompt tokens (2D tensor: [batch, seq_len])
        output_ids: The complete sequence (input + generated tokens) (2D tensor: [batch, total_len])
        **model_kwargs: Additional model arguments (use_cache, output_hidden_states, etc.)
    
    Returns:
        List of model outputs, one per generated token
    """
    IGNORE_KEYS = ["input_ids", "attention_mask", "max_new_tokens", "return_dict_in_generate"]

    return_dict = model_kwargs.get("return_dict_in_generate", True)
    
    # Ensure input_ids has batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)
    
    num_input_tokens = input_ids.shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens
    
    # First pass: process all input tokens to build KV cache
    out = model(
        input_ids,
        past_key_values=None,
        return_dict=return_dict,
        **{k: v for k, v in model_kwargs.items() if k not in IGNORE_KEYS}
    )
    past_key_values = out.past_key_values

    all_outputs = _unpack_input_outputs({k: [v] for k, v in out.items() if k != "past_key_values"})
    
    # Autoregressive generation: process each generated token one at a time
    for t in range(num_input_tokens, num_input_tokens + num_generated_tokens - 1):
        step_out = model(
            input_ids=output_ids[:, t:t+1],
            past_key_values=past_key_values,
            return_dict=return_dict,
            **{
                k: v 
                for k, v in model_kwargs.items() 
                if k not in IGNORE_KEYS
            }
        )
        past_key_values = step_out.past_key_values

        for k, v in step_out.items():
            if k == "past_key_values":
                continue
            all_outputs[k].append(v)
    
    return all_outputs


