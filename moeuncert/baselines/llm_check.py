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

    Implements the LLM-Check scoring suite as individual, independent scores,
    computed at per-token granularity:
    - Attention Score (default): cumulative log-diagonal of attention kernel maps
    - Hidden Score: cumulative mean log of singular values of centered token covariance
    - Perplexity: token-level perplexity
    - Logit Entropy: entropy of the output token distribution

    Per the paper, each score is evaluated independently. The attention score is
    the strongest and most efficient signal.
    """

    def __init__(self, score_type="attention", alpha=0.001, top_k=50, window_size=1):
        """
        Args:
            score_type: Which LLM-Check score to use. One of:
                "attention" — attention cumulative log-diagonal scores (default)
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
    def hidden_score(hidden_states, alpha=0.001):
        """
        Compute per-layer, per-token Hidden Scores.

        Matches the reference implementation centered_svd_val():
        - Transposes hidden states to (d, m) per the paper's convention
        - Centering matrix J is (d, d), centering over hidden dimensions
        - Token covariance Σ = HᵀJH + αI is (m, m)
        - Cumulative score at position t: (1/(t+1)) Σ_{i=1}^{t+1} log σᵢ

        Args:
            hidden_states: Tensor of shape
                (batch_size, seq_len, n_layers, hidden_size).
            alpha: Regularization for covariance matrix.

        Returns:
            Tensor of scores with shape (batch_size, seq_len, n_layers).
        """
        hidden_states = hidden_states.to(torch.float32)
        batch_size, seq_len, n_layers, hidden_size = hidden_states.shape

        # Reshape to merge batch and layers: (B*L, seq_len, hidden_size)
        H = hidden_states.permute(0, 2, 1, 3).reshape(-1, seq_len, hidden_size)

        # Centering over hidden dimensions: J = I - (1/d) 11ᵀ, shape (d, d)
        J = torch.eye(hidden_size, device=H.device) - (1.0 / hidden_size) * torch.ones(
            hidden_size, hidden_size, device=H.device
        )

        # Transpose to (B*L, d, m) — paper's convention
        H_t = H.transpose(-2, -1)

        # Token covariance: H_tᵀ J H_t + αI → (B*L, m, m)
        JH = torch.bmm(J.unsqueeze(0).expand(H_t.shape[0], -1, -1), H_t)  # (B*L, d, m)
        Sigma = torch.bmm(H_t.transpose(-2, -1), JH)  # (B*L, m, m)
        Sigma = Sigma + alpha * torch.eye(seq_len, device=Sigma.device).unsqueeze(0)

        singular_values = torch.linalg.svdvals(Sigma)  # (B*L, m)
        score = torch.cumsum(torch.log(singular_values), dim=-1)  # (B*L, m)
        score /= torch.arange(1, score.shape[-1] + 1, device=score.device)  # normalize

        score = score.reshape(batch_size, seq_len, n_layers)
        return score

    @staticmethod
    def attention_score(attentions):
        """
        Compute per-layer, per-head, per-token Attention Scores.

        For each attention head, computes the cumulative log of the diagonal
        entries of the attention kernel. Per the paper:

            log det(Ker_i) = Σ_{j=1}^{m} log Ker_i[j,j]

        Score at position t is the running log-determinant up to that token.

        Args:
            attentions: Tensor of shape
                (batch_size, n_layers, n_heads, seq_len, seq_len).

        Returns:
            Tensor of attention scores with shape
            (batch_size, n_layers, seq_len).
        """
        # Sum log-diagonals across heads (matching reference: eigscore += ...)
        log_diag = torch.log(attentions.diagonal(dim1=-2, dim2=-1) + 1e-10)  # (B, L, H, m)
        head_sum = log_diag.sum(dim=2)  # (B, L, m)
        return torch.cumsum(head_sum, dim=-1)

    @staticmethod
    def perplexity(scores, input_ids=None):
        """
        Compute perplexity from output logits.

        Args:
            scores: Tensor of output logits with shape
                (batch_size, seq_len, vocab_size).
            input_ids: Tensor of token IDs with shape (batch_size, seq_len).
                If None, uses argmax of logits.

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
        Compute the selected LLM-Check uncertainty score at per-token granularity.

        Args:
            outputs: Dict of standardized model outputs with keys:
                'scores' — (batch_size, seq_len, vocab_size) output logits
                'hidden_states' — (batch_size, seq_len, n_layers, hidden_size)
                'attentions' — (batch_size, n_layers, n_heads, seq_len, seq_len)
                'sequences' — (batch_size, seq_len) token IDs (optional, for perplexity)

        Returns:
            Tensor of uncertainty scores. Shape depends on score_type:
                "attention": (batch_size, n_layers, seq_len)
                "hidden": (batch_size, seq_len, n_layers)
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

        Args:
            outputs: Dict of standardized model outputs.
            labels: Binary hallucination labels (per-token).
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

        # Align labels with score shape by repeating along layer dim if present
        if self.score_type == "attention":
            # probs: (B, n_layers, seq), labels: (B, seq)
            n_layers = probs_np.shape[1]
            probs_flat = probs_np.reshape(-1)
            labels_expanded = np.repeat(labels_np[:, np.newaxis, :], n_layers, axis=1)
            labels_flat = labels_expanded.reshape(-1)
        elif self.score_type == "hidden":
            # probs: (B, seq, n_layers), labels: (B, seq)
            n_layers = probs_np.shape[2]
            probs_flat = probs_np.reshape(-1)
            labels_expanded = np.repeat(labels_np[:, :, np.newaxis], n_layers, axis=2)
            labels_flat = labels_expanded.reshape(-1)
        elif self.score_type == "perplexity":
            # probs: (B,), labels: (B, seq) → reduce labels to answer-level
            probs_flat = probs_np.reshape(-1)
            if labels_np.ndim > 1:
                labels_flat = labels_np.any(axis=-1).astype(float)  # any token hallucinated → 1
            else:
                labels_flat = labels_np.reshape(-1)
        else:
            probs_flat = probs_np.reshape(-1)
            labels_flat = labels_np.reshape(-1)

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
        Predict binary hallucination labels per token.

        Args:
            outputs: Dict of standardized model outputs.

        Returns:
            Binary labels with same shape as predict_proba output.
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict().")

        probs = self.predict_proba(outputs)
        return (probs >= self.threshold).int()