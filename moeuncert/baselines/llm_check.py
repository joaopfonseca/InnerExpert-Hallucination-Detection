"""
LLM-Check baseline (Sriramanan et al., NeurIPS 2024).

A suite of training-free, single-pass hallucination detection methods based on
analyzing internal model representations: hidden state eigenvalue analysis,
attention eigenvalue analysis, and output token uncertainty (perplexity, entropy).

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

    Implements the full LLM-Check scoring suite:
    - Hidden state scores: SVD eigenvalue analysis of hidden state covariance
    - Attention scores: log diagonal of attention kernel similarity maps
    - Perplexity: token-level perplexity of the generated sequence
    - Logit entropy: entropy of the output token distribution

    All scores are computed per-token and aggregated for answer-level detection.
    """

    def __init__(self, score_type="combined", alpha=0.001, top_k=50, window_size=1):
        """
        Args:
            score_type: Which LLM-Check score(s) to use. One of:
                "hidden" — hidden state SVD scores only
                "attention" — attention eigenvalue scores only
                "perplexity" — perplexity only
                "entropy" — logit entropy only
                "combined" — all scores concatenated as features (default)
            alpha: Regularization parameter for SVD covariance matrix.
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
        Compute the SVD-based hidden state score.

        Centers the hidden state matrix, computes the covariance, and returns
        the mean log singular value.

        Args:
            Z: Hidden state tensor of shape (hidden_size, seq_len).
            alpha: Regularization added to covariance diagonal.

        Returns:
            Scalar SVD score.
        """
        # Z shape: (n_tokens, hidden_size)
        # Covariance: Z^T @ J @ Z gives (hidden_size, hidden_size)
        J = torch.eye(Z.shape[0], device=Z.device) - (1 / Z.shape[0]) * torch.ones(
            Z.shape[0], Z.shape[0], device=Z.device
        )
        Sigma = Z.T @ J @ Z
        Sigma = Sigma + alpha * torch.eye(Sigma.shape[0], device=Z.device)
        svdvals = torch.linalg.svdvals(Sigma)
        return torch.log(svdvals).mean()

    @staticmethod
    def hidden_score(hidden_states, alpha=0.001):
        """
        Compute per-layer hidden state SVD scores.

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
                # Z shape: (seq_len, hidden_size)
                Z = hidden_states[b, :, l, :].to(torch.float32)
                layer_scores.append(
                    LLMCheck._centered_svd_score(Z, alpha=alpha).item()
                )
            scores.append(layer_scores)

        return torch.tensor(scores)  # (batch_size, n_layers)

    @staticmethod
    def attention_score(attentions):
        """
        Compute per-layer attention eigenvalue scores.

        For each layer, computes the mean log diagonal of the attention
        matrices across all heads.

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
                layer_scores.append((eigscore / n_heads).item())
            scores.append(layer_scores)

        return torch.tensor(scores)  # (batch_size, n_layers)

    @staticmethod
    def perplexity(scores, input_ids=None):
        """
        Compute per-token perplexity from output logits.

        For each token position, computes the negative log probability of the
        actual generated token, then exponentiates the mean to get perplexity.

        Lower perplexity = more confident = less likely hallucinated.

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

        # Per-token log probabilities
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
        Compute LLM-Check uncertainty scores.

        Args:
            outputs: Dict of standardized model outputs with keys:
                'scores' — (batch_size, seq_len, vocab_size) output logits
                'hidden_states' — (batch_size, seq_len, n_layers, hidden_size)
                'attentions' — (batch_size, n_layers, n_heads, seq_len, seq_len)
                'sequences' — (batch_size, seq_len) token IDs (optional, for perplexity)

        Returns:
            Tensor of uncertainty scores. Shape depends on score_type:
                "hidden" / "attention": (batch_size, n_layers)
                "perplexity": (batch_size,)
                "entropy": (batch_size, seq_len)
                "combined": (batch_size, n_features) — all scores flattened
        """
        if self.score_type == "hidden":
            return self.hidden_score(outputs["hidden_states"], alpha=self.alpha)

        elif self.score_type == "attention":
            return self.attention_score(outputs["attentions"])

        elif self.score_type == "perplexity":
            input_ids = outputs.get("sequences", None)
            return self.perplexity(outputs["scores"], input_ids=input_ids)

        elif self.score_type == "entropy":
            return self.logit_entropy(outputs["scores"], top_k=self.top_k)

        elif self.score_type == "combined":
            features = []

            if "hidden_states" in outputs:
                features.append(
                    self.hidden_score(outputs["hidden_states"], alpha=self.alpha)
                )  # (batch_size, n_layers)

            if "attentions" in outputs:
                features.append(
                    self.attention_score(outputs["attentions"])
                )  # (batch_size, n_layers)

            if "scores" in outputs:
                features.append(
                    self.logit_entropy(
                        outputs["scores"], top_k=self.top_k
                    ).mean(dim=-1, keepdim=True)
                )  # (batch_size, 1)

                input_ids = outputs.get("sequences", None)
                features.append(
                    self.perplexity(
                        outputs["scores"], input_ids=input_ids
                    ).unsqueeze(-1)
                )  # (batch_size, 1)

            return torch.cat(features, dim=-1)  # (batch_size, n_features)

        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    def fit(self, outputs, labels):
        """
        Find the optimal threshold for binary classification.

        For "combined" mode, fits a logistic regression on the feature vector.

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

        if self.score_type == "combined" and probs_np.ndim > 1 and probs_np.shape[1] > 1:
            # Multi-feature: use logistic regression
            from sklearn.linear_model import LogisticRegression

            self._clf = LogisticRegression(max_iter=1000)
            self._clf.fit(probs_np, labels_np)
            self.threshold = 0.5  # default for logistic regression
        else:
            # Single feature: grid search for best threshold
            if probs_np.ndim > 1:
                probs_np = probs_np.mean(axis=-1)
                labels_np = labels_np.mean(axis=-1) if labels_np.ndim > 1 else labels_np

            best_threshold = 0.5
            best_accuracy = 0.0

            for threshold in np.linspace(probs_np.min(), probs_np.max(), 100):
                preds = (probs_np >= threshold).astype(int)
                accuracy = (preds == labels_np).mean()
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold

            self.threshold = best_threshold

    def predict(self, outputs):
        """
        Predict binary hallucination labels.

        Args:
            outputs: Dict of standardized model outputs.

        Returns:
            Binary labels (0 = factual, 1 = hallucinated).
        """
        if self.threshold is None and not hasattr(self, "_clf"):
            raise RuntimeError("Must call fit() before predict().")

        probs = self.predict_proba(outputs)

        if hasattr(self, "_clf"):
            if isinstance(probs, torch.Tensor):
                probs_np = probs.cpu().numpy()
            else:
                probs_np = np.array(probs)
            preds = self._clf.predict(probs_np)
            return torch.tensor(preds, dtype=torch.int)
        else:
            return (probs >= self.threshold).int()