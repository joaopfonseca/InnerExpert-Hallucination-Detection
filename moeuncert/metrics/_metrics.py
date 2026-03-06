import torch


def hidden_score(hidden_states):
    """
    Compute hidden state scores based on the LLM-Check method.

    Args:
        hidden_states: Tensor of hidden states with shape
        (batch_size, sequence_length, n_layers, hidden_size)

    Returns:
        Tensor of hidden state scores with shape (batch_size, sequence_length, n_layers)
    """

    hidden_states_transposed = torch.transpose(
        hidden_states, dim0=-2, dim1=-1
    )  # (batch_size, sequence_length, hidden_size, n_layers)
    cov_matrices = (
        hidden_states @ hidden_states_transposed
    )  # (batch_size, sequence_length, n_layers, n_layers)
    singular_values = torch.linalg.svd(
        cov_matrices.to(torch.float32)
    ).S  # (batch_size, sequence_length, n_layers)
    score = 2 * torch.cumsum(torch.log(singular_values), dim=-1)
    score /= torch.arange(score.shape[-1]) + 1

    return score


def attention_score(attentions):
    """
    Compute attention scores based on the LLM-Check method.

    Args:
        attentions: Tensor of attention matrices with shape
        ([batch_size, ]num_layers, num_heads, seq_length, seq_length)

    Returns:
        Tensor of attention scores with shape (num_layers, seq_length)
    """
    score = torch.cumsum(torch.log(attentions.diagonal(dim1=-2, dim2=-1)), dim=-1)
    return score


def topk_entropy(scores, k=None, softmax=True):
    """
    Compute the top-k entropy of the output logits.

    Args:
        scores: Tensor of output logits with shape (batch_size, sequence_length, vocab_size)
        k: Number of top tokens to consider for entropy calculation

    Returns:
        Tensor of top-k entropy scores with shape (batch_size, sequence_length)
    """

    if softmax:
        scores = torch.softmax(scores, dim=-1)  # (batch_size, sequence_length, k)

    if k is not None:
        scores, _ = torch.topk(scores, k=k, dim=-1)  # (batch_size, sequence_length, k)

    entropy = -torch.sum(scores * torch.log(scores + 1e-10), dim=-1)  # (batch_size, sequence_length)
    return entropy


def cosine_similarity(hidden_states, eps=1e-08):
    """
    Compute the cosine similarity among expert hidden states.

    Args:
        hidden_states: Tensor of expert hidden states with shape
        (batch_size, sequence_length, n_layers, n_experts, hidden_size)

    Returns:
        Tensor of cosine similarity scores with shape (batch_size,
        sequence_length, n_layers, n_experts, n_experts)
    """
    norms = hidden_states.norm(dim=-1, keepdim=True).clamp_min(eps)
    hidden_states_norm = hidden_states / norms  # (B, S, L, E, H)

    # reshape to merge leading dims, then matmul, then reshape back
    shape = hidden_states_norm.shape  # (B, S, L, E, H)
    hsv = hidden_states_norm.reshape(-1, shape[-2], shape[-1])  # (BSL, E, H)
    sim = torch.matmul(hsv, hsv.transpose(1, 2))  # (BSL, E, E)
    sim = sim.reshape(*shape[:-1], shape[-2])  # (B, S, L, E, E)
    return sim
