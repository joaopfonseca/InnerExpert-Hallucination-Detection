import torch
import torch.nn.functional as F


def hidden_score(hidden_states, alpha=0.001):
    """
    Compute hidden state scores based on the LLM-Check method.

    Computes the mean log-determinant of the centered covariance matrix of
    hidden states, following the LLM-Check implementation (Sriramanan et al.,
    NeurIPS 2024).

    The paper defines: Σ² = HᵀH where H is (d × m), giving (m × m) token covariance.
    The implementation adds centering over the hidden dimension (J = I - (1/d) 11ᵀ)
    and αI regularization: Σ = Hᵀ J H + αI.

    The centering is over hidden features (J is d×d), which is different from
    centering over tokens. This matches the official codebase's centered_svd_val().

    Note: The paper explicitly contrasts this with INSIDE (Chen et al., 2024),
    which computes a centered covariance across *multiple model responses*.

    Args:
        hidden_states: Tensor of hidden states with shape
        (batch_size, sequence_length, n_layers, hidden_size)
        alpha: Regularization parameter added to the covariance diagonal.
            Defaults to 0.001, matching the LLM-Check codebase.

    Returns:
        Tensor of hidden state scores with shape (batch_size, sequence_length, n_layers)
    """
    hidden_states = hidden_states.to(torch.float32)
    batch_size, seq_len, n_layers, hidden_size = hidden_states.shape

    # Reshape to merge batch and layers: (B*L, seq_len, hidden_size)
    H = hidden_states.permute(0, 2, 1, 3).reshape(-1, seq_len, hidden_size)

    # Centering matrix over hidden dimension: J = I - (1/d) 11ᵀ, shape (d, d)
    J = torch.eye(hidden_size, device=H.device) - (1.0 / hidden_size) * torch.ones(
        hidden_size, hidden_size, device=H.device
    )

    # Transpose H to (d, m) per the paper's convention, then: Σ = Hᵀ J H + αI → (m, m)
    # H is (B*L, seq_len, d) → H_t is (B*L, d, seq_len)
    H_t = H.transpose(-2, -1)

    # H_t is (B*L, d, m). J is (d, d).
    # J H_t → (B*L, d, m). Then H_tᵀ (J H_t) → (B*L, m, m)
    JH = torch.bmm(J.unsqueeze(0).expand(H_t.shape[0], -1, -1), H_t)  # (B*L, d, m)
    Sigma = torch.bmm(H_t.transpose(-2, -1), JH)  # (B*L, m, m)
    Sigma = Sigma + alpha * torch.eye(seq_len, device=Sigma.device).unsqueeze(0)

    singular_values = torch.linalg.svdvals(Sigma)  # (B*L, m)
    score = torch.cumsum(torch.log(singular_values), dim=-1)  # (B*L, m)
    score /= torch.arange(1, score.shape[-1] + 1, device=score.device)  # normalize by position

    score = score.reshape(batch_size, seq_len, n_layers)
    return score


def attention_score(attentions):
    """
    Compute attention scores based on the LLM-Check method.

    For each attention head, computes the log of the diagonal entries of the
    attention kernel similarity map. Per the paper, the eigenvalues of the
    lower-triangular attention kernel are exactly the diagonal entries, so
    the log-determinant is:

        log det(Ker_i) = Σ_{j=1}^{m} log Ker_i[j,j]

    We compute this cumulatively so that the score at position t represents
    the running log-determinant up to that token. This preserves per-token
    granularity while matching the paper's formula.

    The paper aggregates these into a single scalar per layer by summing
    across heads and averaging over positions. Here we keep per-head and
    per-token granularity for token-level hallucination detection.

    Args:
        attentions: Tensor of attention matrices with shape
        ([batch_size, ]num_layers, num_heads, seq_length, seq_length)

    Returns:
        Tensor of attention scores with shape (batch_size, num_layers, num_heads, seq_length)
    """
    return torch.cumsum(torch.log(attentions.diagonal(dim1=-2, dim2=-1) + 1e-10), dim=-1)


def topk_entropy(scores, k=None, softmax=True):
    """
    Compute the top-k entropy of the output logits.

    Args:
        scores: Tensor of output logits with shape (batch_size, sequence_length, vocab_size)
        k: Number of top scores to consider for entropy calculation. If None,
        considers all scores.

    Returns:
        Tensor of top-k entropy scores with shape (batch_size, sequence_length)
    """

    if softmax:
        scores = torch.softmax(scores, dim=-1)  # (batch_size, sequence_length, k)

    if k is not None:
        scores, _ = torch.topk(scores, k=k, dim=-1)  # (batch_size, sequence_length, k)

    entropy = -torch.sum(
        scores * torch.log(scores + 1e-10), dim=-1
    )  # (batch_size, sequence_length)
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


def expert_hidden_score(expert_hidden_states, expert_weights):
    """
    Compute a weighted hidden state score across experts per layer.

    Computes the hidden state score (via `hidden_score`) for each expert and
    aggregates them into a single score per layer using the routing weights as
    a weighted average.

    Args:
        expert_hidden_states: Tensor of expert hidden states with shape
            (batch_size, sequence_length, n_layers, n_experts, hidden_size)
        expert_weights: Tensor of routing weights with shape
            (batch_size, sequence_length, n_layers, n_experts)

    Returns:
        Tensor of weighted hidden state scores with shape
        (batch_size, sequence_length, n_layers)
    """
    # hidden_score expects a 4D tensor (B, S, L, H). Score each expert independently.
    batch_size, seq_len, n_layers, n_experts, hidden_size = expert_hidden_states.shape
    expert_hidden_states = (
        expert_hidden_states
        .permute(0, 3, 1, 2, 4)  # (B, E, S, L, H)
        .reshape(
            batch_size * n_experts, seq_len, n_layers, hidden_size
        )  # (B*E, S, L, H)
    )
    expert_hidden_scores = hidden_score(expert_hidden_states)  # (B*E, S, L)

    expert_hidden_scores = expert_hidden_scores.reshape(
        batch_size, n_experts, seq_len, n_layers
    ).permute(0, 2, 3, 1)  # (B, S, L, E)

    expert_weights = expert_weights / expert_weights.sum(
        dim=-1, keepdim=True
    )  # Normalize weights
    expert_hidden_scores = (expert_hidden_scores * expert_weights).sum(
        dim=-1
    )  # Sum over experts
    return expert_hidden_scores


def expert_similarity_score(expert_hidden_states, expert_weights):
    """
    Compute a routing-weight-weighted average of pairwise cosine similarities between experts.

    For each token position and layer, computes the full `(n_experts, n_experts)` cosine
    similarity matrix and takes a weighted sum using the outer product of the routing
    weights, producing a scalar similarity score. Higher values indicate that the
    activated experts produced more similar hidden states.

    Args:
        expert_hidden_states: Tensor of expert hidden states with shape
            (batch_size, sequence_length, n_layers, n_experts, hidden_size)
        expert_weights: Tensor of routing weights with shape
            (batch_size, sequence_length, n_layers, n_experts)

    Returns:
        Tensor of weighted expert similarity scores with shape
        (batch_size, sequence_length, n_layers)
    """
    expert_similarities = cosine_similarity(expert_hidden_states)
    expert_weights = expert_weights / expert_weights.sum(
        dim=-1, keepdim=True
    )  # Normalize weights
    expert_weights = expert_weights.view(*expert_weights.shape, 1)
    expert_weights = expert_weights @ expert_weights.transpose(-1, -2)
    expert_similarities = (expert_similarities * expert_weights).sum(dim=(-2, -1))
    return expert_similarities


def expert_usage_frequency(expert_idx, weights=None):
    """
    Compute cumulative expert usage proportions across the sequence.

    Returns a tensor where each position represents the cumulative proportion
    of times each expert has been selected up to that token, normalized by
    the cumulative total selections so far. This captures how the routing
    distribution evolves over the generated sequence.

    Args:
        expert_idx: Integer tensor of selected expert indices with shape
            (batch_size, sequence_length, n_layers, top_k)
        weights: Optional tensor of routing weights with shape
            (batch_size, sequence_length, n_layers, top_k) to compute a
            weighted usage score instead of raw frequency

    Returns:
        Float tensor of cumulative expert usage proportions with shape
        (batch_size, sequence_length, n_layers, n_experts)
    """
    expert_usage = torch.nn.functional.one_hot(
        expert_idx, num_classes=expert_idx.max() + 1
    ).sum(
        # dim=(1, 3)  # sums over tokens and selected-experts
        dim=3  # sum over selected-experts only, to get usage per token position
    )  # (batch_size, sequence_length, n_layers, n_experts)

    if weights is not None:
        # If weights are provided, compute a weighted usage score
        # Expand weights to match one-hot shape: (batch, seq, layers, top_k, n_experts)
        one_hot = torch.nn.functional.one_hot(
            expert_idx, num_classes=expert_idx.max() + 1
        )  # (batch, seq, layers, top_k, n_experts)
        weights_expanded = weights.unsqueeze(-1)  # (batch, seq, layers, top_k, 1)
        expert_usage = (one_hot * weights_expanded).sum(dim=3)  # (batch, seq, layers, n_experts)

    expert_usage = torch.cumsum(
        expert_usage, dim=1, dtype=torch.float32
    )  # cumulative absolute usage over sequence

    expert_usage = (
        expert_usage / expert_usage.sum(-1, keepdim=True)
    )  # normalize by token position
    return expert_usage


def expert_usage_gini_impurity(expert_usage):
    """
    Compute Gini impurity from expert usage probabilities.

    Args:
        expert_usage: Tensor of expert usage probabilities with shape
            (batch_size, sequence_length, n_layers, n_experts)

    Returns:
        Tensor of Gini impurity scores with shape
        (batch_size, sequence_length, n_layers)
    """
    return 1.0 - torch.sum(expert_usage**2, dim=-1)


def inverse_herfindahl_index(expert_usage, eps=1e-10):
    """
    Compute number of effective experts from expert usage probabilities.

    This is the inverse Herfindahl index: 1 / sum(p_i^2).

    Args:
        expert_usage: Tensor of expert usage probabilities with shape
            (batch_size, sequence_length, n_layers, n_experts)
        eps: Small constant for numerical stability

    Returns:
        Tensor of effective-expert counts with shape
        (batch_size, sequence_length, n_layers)
    """
    return 1.0 / torch.sum(expert_usage**2 + eps, dim=-1)


def compute_metrics(standardized_outputs, return_baseline_features=False):
    """Compute uncertainty metrics from standardized model outputs.

    Args:
        standardized_outputs: Dict with keys like 'hidden_states', 'attentions',
            'scores', 'sequences', etc. from standardize_outputs().
        return_baseline_features: If True, also compute per-token log-likelihoods
            and full-vocabulary entropies needed by trainable baselines like
            HaluNet. Default False to avoid unnecessary computation.

    Returns:
        Dict of computed metrics.
    """

    metrics = {}
    if "hidden_states" in standardized_outputs:
        # Shape of hidden states: (batch_size, sequence_length, n_layers, hidden_size)
        # Shape of hidden scores: (batch_size, sequence_length, n_layers)
        metrics["hidden_scores"] = hidden_score(standardized_outputs["hidden_states"])

    if "attentions" in standardized_outputs:
        # Shape of attention matrices: (batch_size, n_layers, num_heads, seq_len, seq_len)
        # Shape of attention scores: (batch_size, n_layers, num_heads, seq_len)
        metrics["attention_scores"] = attention_score(
            standardized_outputs["attentions"]
        )

    if "scores" in standardized_outputs:
        # Shape of output logits: (batch_size, sequence_length, vocab_size)
        # Shape of entropy scores: (batch_size, sequence_length)
        # Full-vocabulary entropy (k=None) for PredictiveEntropy correctness
        metrics["scores_entropy"] = topk_entropy(standardized_outputs["scores"])

        # Baseline-specific features for trainable methods (e.g., HaluNet)
        if return_baseline_features:
            scores = standardized_outputs["scores"]  # (B, gen_seq_len, vocab_size)
            sequences = standardized_outputs["sequences"]  # (B, seq_len)
            gen_seq_len = scores.shape[1]

            # Per-token log-likelihoods: log p(x_t | x_{<t}) for each generated token
            log_probs = F.log_softmax(scores, dim=-1)  # (B, gen_seq_len, vocab_size)
            # Gather log prob of the actual generated token
            gen_token_ids = sequences[:, -gen_seq_len:].unsqueeze(-1)  # (B, gen_seq_len, 1)
            log_likelihoods = log_probs.gather(-1, gen_token_ids).squeeze(-1)  # (B, gen_seq_len)
            metrics["log_likelihoods"] = log_likelihoods

            # Per-token full-vocabulary entropies: H_t = -Σ_v p(v) log p(v)
            probs = F.softmax(scores, dim=-1)  # (B, gen_seq_len, vocab_size)
            entropies = -(probs * log_probs).sum(dim=-1)  # (B, gen_seq_len)
            metrics["entropies"] = entropies

            # Answer-level perplexity: exp(-mean(log p(x_t | x_{<t})))
            metrics["perplexity"] = torch.exp(-log_likelihoods.mean(dim=1))  # (B,)

            # Last-layer hidden states for HaluNet embedding branch
            # hidden_states: (B, full_seq_len, n_layers, hidden_size)
            # Slice to generated positions only, keep only the last layer
            if "hidden_states" not in standardized_outputs:
                raise KeyError(
                    "'hidden_states' required when return_baseline_features=True. "
                    "Ensure the model was loaded with output_hidden_states=True."
                )
            gen_hidden = standardized_outputs["hidden_states"][:, -gen_seq_len:, -1, :]
            metrics["last_hidden_states"] = gen_hidden  # (B, gen_seq_len, hidden_dim)

    if "expert_weights" in standardized_outputs:
        # Shape of expert weights: (batch_size, sequence_length, n_layers, n_experts)
        # Shape of router entropy scores: (batch_size, sequence_length, n_layers)
        # NOTE: expert weights do not sum to 1 if
        #       model.model.layers[...].mlp.norm_topk_prob is False
        metrics["router_entropy"] = topk_entropy(
            standardized_outputs["expert_weights"], softmax=False
        )

    if (
        "expert_hidden_states" in standardized_outputs
        and "expert_weights" in standardized_outputs
    ):
        # Shape of expert hidden states:
        # (batch_size, sequence_length, n_layers, n_experts, hidden_size)
        # Option 1: sum hidden state score over experts, weighing by the expert weights
        metrics["expert_hidden_scores"] = expert_hidden_score(
            standardized_outputs["expert_hidden_states"],
            standardized_outputs["expert_weights"],
        )

        # Option 2: weighted sum of cosine similarity among expert hidden states
        metrics["expert_similarities"] = expert_similarity_score(
            standardized_outputs["expert_hidden_states"],
            standardized_outputs["expert_weights"],
        )

    if "expert_idx" in standardized_outputs:
        # Check usage frequency of each expert
        metrics["expert_usage"] = expert_usage_frequency(
            standardized_outputs["expert_idx"],
            weights=standardized_outputs["expert_weights"],
            # [:, standardized_outputs["input_ids"].shape[-1]:]
        )
        metrics["expert_usage_entropy"] = topk_entropy(
            metrics["expert_usage"], softmax=False
        )
        metrics["expert_usage_gini"] = expert_usage_gini_impurity(
            metrics["expert_usage"]
        )
        metrics["expert_usage_effective_experts"] = inverse_herfindahl_index(
            metrics["expert_usage"]
        )

    return metrics
