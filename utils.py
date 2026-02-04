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
    attentions = outputs.attentions
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
