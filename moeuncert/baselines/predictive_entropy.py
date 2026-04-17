"""
Predictive Entropy baseline.

Computes token-level entropy from the model's output probability distribution.
The simplest uncertainty baseline — if a method can't beat this, it's not useful.

No specific paper to cite; used as the standard floor baseline in virtually all
LLM uncertainty quantification work (e.g., Kuhn et al., 2023; Yadkori et al.,
2024).
"""

import torch


def predictive_entropy(scores, softmax=True):
    """
    Compute per-token predictive entropy from model output logits.

    Predictive entropy measures the uncertainty in the model's token-level
    probability distribution. High entropy means the model is unsure which
    token comes next; low entropy means it's confident.

    This is the simplest uncertainty baseline and serves as a floor for
    any more sophisticated method.

    Args:
        scores: Tensor of output logits with shape
            (batch_size, sequence_length, vocab_size)
        softmax: If True, apply softmax to convert logits to probabilities
            before computing entropy. If False, assume scores are already
            probabilities.

    Returns:
        Tensor of per-token predictive entropy scores with shape
        (batch_size, sequence_length)
    """
    if softmax:
        probs = torch.softmax(scores, dim=-1)
    else:
        probs = scores

    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy