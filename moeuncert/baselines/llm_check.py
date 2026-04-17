"""
LLM-Check baseline (Sriramanan et al., NeurIPS 2024).

A suite of training-free, single-pass hallucination detection methods based on
analyzing internal model representations. Each score is evaluated independently
per the original paper — they do not combine scores into a single classifier.

Paper: LLM-Check: Investigating Detection of Hallucinations in Large Language Models
       (NeurIPS 2024)
Code: https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection
"""

import torch
import numpy as np
from ._base import BaseBaseline


class LLMCheck(BaseBaseline):
    """
    LLM-Check baseline for hallucination detection.

    Implements the LLM-Check scoring suite as individual, independent scores:
    - Attention Score (default): sum of mean log-diagonal of attention kernel maps
    - Hidden Score: mean log of singular values of centered token covariance
    - Perplexity: token-level perplexity
    - Logit Entropy: entropy of the output token distribution

    Per the paper, each score is evaluated independently — the authors found
    the Attention Score to be the strongest and most efficient signal.
    """

    def __init__(self, score_type="attention", alpha=0.001, top_k=50, window_size=1):
        """
        Args:
            score_type: Which LLM-Check score to use. One of:
                "attention" — attention eigenvalue scores (default, strongest per paper)
                "hidden" — hidden state SVD scores
                "perplexity" — perplexity
                "entropy" — logit entropy
            alpha: Regularization parameter for hidden state SVD covariance matrix.
            top_k: Number of top tokens for logit entropy. None for full vocab.
            window_size: Window size for sliding-window entropy.
        """
        self.score_type = score_type
        self.alpha = alpha
        self.top_k = top_k
        self.window_size = window_size
        self.threshold = None

    @staticmethod
    def _centered_svd_score(Z, alpha=0.001):
        """
        Compute the SVD-based hidden state score (Hidden Score).

        Matches the reference implementation centered_svd_val():
        - Z is transposed to (d, m) where d = hidden_size, m = seq_len
        - Centering matrix J is (d, d), centering over hidden dimensions
        - Covariance Σ = Zᵀ J Z is (m, m) — token covariance
        - Score = mean(log(σᵢ)) where σᵢ are singular values of Σ

        Args:
            Z: Hidden state tensor of shape (seq_len, hidden_size).
            alpha: Regularization added to covariance diagonal.

        Returns:
            Scalar Hidden Score.
        """
        # Transpose to (d, m) — matching reference's Z = torch.transpose(Z, 0, 1)
        Z = Z.T  # (d, m)

        # Centering over hidden dimensions: J = I - (1/d) 11ᵀ, shape (d, d)
        J = torch.eye(Z.shape[0], device=Z.device) - (1 / Z.shape[0]) * torch.ones(
            Z.shape[0], Z.shape[0], device=Z.device
        )

        # Token covariance: Zᵀ J Z → (m, d) @ (d, d) @ (d, m) = (m, m)
        Sigma = Z.T @ J @ Z
        Sigma = Sigma + alpha * torch.eye(Sigma.shape[0], device=Z.device)
        svdvals = torch.linalg.svdvals(Sigma)
        # Reference uses log(svdvals).mean() — no factor of 2
        return torch.log(svdvals).mean()

    @staticmethod
    def hidden_score(hidden_states, alpha=0.001):
        """
        Compute per-layer Hidden Scores.

        The mean log of singular values of the centered token covariance matrix
        at each layer, following the reference implementation's centered_svd_val().

        Args:
            hidden_states: Tensor of shape
                (batch_size, seq_len, n_layers, hidden_size).
            alpha: Regularization for covariance matrix.

        Returns:
            Tensor of per-layer scores with shape (batch_size, n_layers).
        """
        batch_size, seq_len, n_layers, hidden_size = hidden_states.shape
        scores = []

        for b in range(batch_size):
            layer_scores = []
            for l in range(n_layers):
                Z = hidden_states[b, :, l, :].to(torch.float32)  # (seq_len, hidden_size)
                layer_scores.append(
                    LLMCheck._centered_svd_score(Z, alpha=alpha).item()
                )
            scores.append(layer_scores)

        return torch.tensor(scores)  # (batch_size, n_layers)

    @staticmethod
    def attention_score(attentions):
        """
        Compute per-layer Attention Scores.

        For each layer, computes the sum of mean log-diagonal of the attention
        kernel similarity maps across all heads. This matches the reference:

            eigscore += torch.log(torch.diagonal(Sigma, 0)).mean()

        summed over heads (not averaged).

        Args:
            attentions: Tensor of shape
                (batch_size, n_layers, n_heads, seq_len, seq_len).

        Returns:
            Tensor of per-layer scores with shape (batch_size, n_layers).
        """
        batch_size, n_layers, n_heads, seq_len, _ = attentions.shape
        scores = []

        for b in range(batch_size):
            layer_scores = []
            for l in range(n_layers):
                eigscore = 0.0
                for h in range(n_heads):
                    attn = attentions[b, l, h]  # (seq_len, seq_len)
                    eigscore += torch.log(
                        torch.diagonal(attn, 0) + 1e-10
                    ).mean()
                layer_scores.append(eigscore.item())
            scores.append(layer_scores)

        return torch.tensor(scores)  # (batch_size, n_layers)

    @staticmethod
    def perplexity(scores, input_ids=None):
        """
        Compute perplexity from output logits.

        Args:
            scores: Tensor of output logits with shape
                (batch_size, seq_len, vocab_size).
            input_ids: Tensor of token IDs with shape (batch_size, seq_len).
                If None, uses argmax of logits (teacher forcing with predicted tokens).

        Returns:
            Tensor of perplexity scores with shape (batch_size,).
        """
        probs = torch.softmax(scores, dim=-1)

        if input_ids is None:
            input_ids = scores.argmax(dim=-1)

        log_probs = torch.log(probs + 1e-10)
        token_log_probs = log_probs.gather(
            2, input_ids.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, seq_len)

        ppls = torch.exp(-token_log_probs.mean(dim=-1))  # (batch_size,)
        return ppls

    @staticmethod
    def logit_entropy(scores, top_k=None):
        """
        Compute per-token logit entropy.

        Args:
            scores: Tensor of output logits with shape
                (batch_size, seq_len, vocab_size).
            top_k: Number of top tokens to consider. None for full vocab.

        Returns:
            Tensor of per-token entropy with shape (batch_size, seq_len).
        """
        probs = torch.softmax(scores, dim=-1)

        if top_k is not None:
            probs, _ = torch.topk(probs, k=top_k, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy  # (batch_size, seq_len)

    def predict_proba(self, outputs):
        """
        Compute the selected LLM-Check uncertainty score.

        Args:
            outputs: Dict of standardized model outputs with keys:
                'scores' — (batch_size, seq_len, vocab_size) output logits
                'hidden_states' — (batch_size, seq_len, n_layers, hidden_size)
                'attentions' — (batch_size, n_layers, n_heads, seq_len, seq_len)
                'sequences' — (batch_size, seq_len) token IDs (optional, for perplexity)

        Returns:
            Tensor of uncertainty scores. Shape depends on score_type:
                "attention": (batch_size, n_layers)
                "hidden": (batch_size, n_layers)
                "perplexity": (batch_size,)
                "entropy": (batch_size, seq_len)
        """
        if self.score_type == "attention":
            return self.attention_score(outputs["attentions"])

        elif self.score_type == "hidden":
            return self.hidden_score(outputs["hidden_states"], alpha=self.alpha)

        elif self.score_type == "perplexity":
            input_ids = outputs.get("sequences", None)
            return self.perplexity(outputs["scores"], input_ids=input_ids)

        elif self.score_type == "entropy":
            return self.logit_entropy(outputs["scores"], top_k=self.top_k)

        else:
            raise ValueError(f"Unknown score_type: {self.score_type}. "
                             "Use 'attention', 'hidden', 'perplexity', or 'entropy'.")

    def fit(self, outputs, labels):
        """
        Find the optimal threshold for binary classification.

        Since each score type is used independently per the paper, we find
        the best threshold via grid search on the score values.

        Args:
            outputs: Dict of standardized model outputs.
            labels: Binary hallucination labels.
        """
        probs = self.predict_proba(outputs)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy().astype(float)
        else:
            labels_np = np.array(labels, dtype=float)

        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = np.array(probs)

        # Flatten to 1D if needed (e.g., per-layer scores → take mean across layers)
        if probs_np.ndim > 1:
            probs_flat = probs_np.mean(axis=-1)
        else:
            probs_flat = probs_np

        if labels_np.ndim > 1:
            labels_flat = labels_np.mean(axis=-1)
        else:
            labels_flat = labels_np

        # Grid search for best threshold (maximize accuracy)
        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in np.linspace(probs_flat.min(), probs_flat.max(), 100):
            preds = (probs_flat >= threshold).astype(int)
            accuracy = (preds == labels_flat).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold

    def predict(self, outputs):
        """
        Predict binary hallucination labels using the fitted threshold.

        Args:
            outputs: Dict of standardized model outputs.

        Returns:
            Binary labels (0 = factual, 1 = hallucinated).
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict().")

        probs = self.predict_proba(outputs)

        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = np.array(probs)

        if probs_np.ndim > 1:
            probs_flat = probs_np.mean(axis=-1)
        else:
            probs_flat = probs_np

        preds = (probs_flat >= self.threshold).astype(int)
        return torch.tensor(preds, dtype=torch.int)