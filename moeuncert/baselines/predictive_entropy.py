"""
Predictive Entropy baseline.

Computes token-level entropy from the model's output probability distribution.
The simplest uncertainty baseline — if a method can't beat this, it's not useful.

No specific paper to cite; used as the standard floor baseline in virtually all
LLM uncertainty quantification work (e.g., Kuhn et al., 2023; Yadkori et al.,
2024).
"""

import torch
import numpy as np
from ._base import BaseBaseline


class PredictiveEntropy(BaseBaseline):
    """
    Predictive entropy baseline for hallucination detection.

    Computes per-token entropy from the model's output probability distribution.
    High entropy indicates uncertainty in the model's next-token prediction,
    which may signal hallucination.
    """

    def __init__(self):
        self.threshold = None

    def fit(self, scores, labels):
        """
        Find the optimal threshold for binary classification.

        Args:
            scores: Tensor of output logits with shape
                (batch_size, sequence_length, vocab_size)
            labels: Binary hallucination labels with shape
                (batch_size, sequence_length) or (batch_size,)
        """
        probs = self.predict_proba(scores)

        # Find threshold that maximizes accuracy
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)

        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = np.array(probs)

        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in np.linspace(probs_np.min(), probs_np.max(), 100):
            preds = (probs_np >= threshold).astype(int)
            accuracy = (preds == labels_np).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold

    def predict(self, scores):
        """
        Predict binary hallucination labels using the fitted threshold.

        Args:
            scores: Tensor of output logits with shape
                (batch_size, sequence_length, vocab_size)

        Returns:
            Binary labels tensor (0 = factual, 1 = hallucinated).
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict().")

        uncertainty = self.predict_proba(scores)
        return (uncertainty >= self.threshold).int()

    def predict_proba(self, scores, softmax=True):
        """
        Compute per-token predictive entropy as uncertainty scores.

        Args:
            scores: Tensor of output logits with shape
                (batch_size, sequence_length, vocab_size)
            softmax: If True, apply softmax to convert logits to probabilities
                before computing entropy. If False, assume scores are already
                probabilities.

        Returns:
            Tensor of per-token entropy scores with shape
            (batch_size, sequence_length)
        """
        if softmax:
            probs = torch.softmax(scores, dim=-1)
        else:
            probs = scores

        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy