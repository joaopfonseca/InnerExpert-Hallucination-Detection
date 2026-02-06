import torch


def hidden_score(hidden_states):
    """
    Compute hidden state scores based on the LLM-Check method.

    Args:
        hidden_states: Tensor of hidden states with shape
        (num_layers + 1, seq_length, hidden_size)

    Returns:
        Tensor of hidden state scores with shape (num_layers + 1, seq_length)
    """

    hidden_states_transposed = torch.transpose(
        hidden_states, dim0=-2, dim1=-1
    )  # (batch_size, num_layers, hidden_size, seq_length)
    cov_matrices = (
        hidden_states @ hidden_states_transposed
    )  # (batch_size, num_layers, seq_length, seq_length)
    singular_values = torch.linalg.svd(
        cov_matrices
    ).S  # (batch_size, num_layers, seq_length)
    score = 2 * torch.cumsum(torch.log(singular_values), dim=-1)
    score /= torch.arange(score.shape[-1]) + 1

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
