"""
Token-Level Mahalanobis Distance baseline (Vazhentsev et al., 2025).

Extracts token embeddings from multiple decoder layers, computes Mahalanobis
distance per token per layer, and trains a linear regression on PCA'd features
to produce uncertainty scores.

Paper: Token-Level Density-Based Uncertainty Quantification Methods for
       Eliciting Truthfulness of Large Language Models (arXiv 2502.14427)
Code: https://github.com/ArtemVazh/token_mahalanobis_distance

Key idea: Density-based UQ methods, originally designed for classification
OOD detection, are adapted to text generation. For each token, the Mahalanobis
distance measures how far its hidden state is from the "correct" (factual)
embedding distribution across layers. High MD → token is atypical → likely
hallucinated.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ._base import BaseBaseline


class TokenMahalanobis(BaseBaseline):
    """Token-Level Mahalanobis Distance baseline.

    Computes per-token Mahalanobis distance using hidden states from
    multiple decoder layers, then trains a Ridge regression on the
    PCA-reduced layer-wise MD features (optionally augmented with
    sequence-level log-probability) to predict hallucination scores.

    Args:
        n_components: Number of PCA components for layer-wise MD features.
            If None, use min(n_layers, 10). Default None.
        alpha: Ridge regression regularization strength. Default 1.0.
        use_logprob: Whether to include sequence log-probability as an
            additional feature per the paper. Default True.
        handle_nan: If True, replace inf/nan values with column mean.
            Default True.
    """

    def __init__(self, n_components=None, alpha=1.0, use_logprob=True,
                 handle_nan=True):
        self.n_components = n_components
        self.alpha = alpha
        self.use_logprob = use_logprob
        self.handle_nan = handle_nan

        self.pca = None
        self.scaler = None
        self.regressor = None
        self.class_means_ = None          # list of per-layer class-0 means
        self.shared_cov_inv_ = None       # list of per-layer shared cov inverses
        self.eps_ = 1e-6
        self.feature_dim_ = None
        self.threshold_ = 0.5

    def _safe_replace(self, arr, fill_value=0.0):
        """Replace non-finite values with fill_value."""
        return np.where(np.isfinite(arr), arr, fill_value)

    def _compute_mahalanobis_per_layer(self, embeddings, class_means,
                                        cov_inv_list):
        """Compute per-layer Mahalanobis distance.

        Args:
            embeddings: (n_tokens, n_layers, hidden_size)
            class_means: list of (hidden_size,) arrays, one per layer
            cov_inv_list: list of (hidden_size, hidden_size) arrays

        Returns:
            (n_tokens, n_layers) array of MD values.
        """
        n_tokens, n_layers, hidden_size = embeddings.shape
        md_per_layer = np.zeros((n_tokens, n_layers))

        for layer_idx in range(n_layers):
            layer_emb = embeddings[:, layer_idx, :]  # (n_tokens, hidden_size)
            centered = layer_emb - class_means[layer_idx]  # (n_tokens, hidden_size)
            # Mahalanobis distance: sqrt(centered @ cov_inv @ centered.T)
            # Compute per-row: (x-mu)^T Sigma^{-1} (x-mu)
            try:
                left = centered @ cov_inv_list[layer_idx]  # (n_tokens, hidden_size)
                md = np.sum(left * centered, axis=1)        # (n_tokens,)
                md = np.sqrt(np.maximum(md, 0.0))           # guard against tiny negatives
            except Exception:
                # Fallback: compute sequentially if matrix ops fail
                md = np.zeros(n_tokens)
                for i in range(n_tokens):
                    try:
                        md[i] = np.sqrt(
                            centered[i] @ cov_inv_list[layer_idx] @ centered[i]
                        )
                    except Exception:
                        md[i] = 0.0
            md_per_layer[:, layer_idx] = md

        return md_per_layer

    def _estimate_covariance(self, embeddings_correct):
        """Estimate a regularized shared covariance matrix.

        Args:
            embeddings_correct: (n_correct_tokens, hidden_size)

        Returns:
            (hidden_size, hidden_size) — regularized covariance inverse.
        """
        n = embeddings_correct.shape[0]
        if n < 2:
            # Not enough data; return identity * small epsilon
            hidden_size = embeddings_correct.shape[1]
            return np.eye(hidden_size) / self.eps_

        centered = embeddings_correct - embeddings_correct.mean(axis=0)
        cov = (centered.T @ centered) / (n - 1)

        # Regularize: shrink towards diagonal for numerical stability
        # Use shrinkage: (1 - rho) * cov + rho * trace(cov)/d * I
        d = cov.shape[0]
        trace_cov = np.trace(cov)
        rho = min(0.1, 1.0 / max(n, 1))  # stronger shrinkage for small n
        cov_reg = (1 - rho) * cov + rho * (trace_cov / d) * np.eye(d)

        # Pseudoinverse for numerical stability
        try:
            cov_inv = np.linalg.pinv(cov_reg)
        except Exception:
            cov_inv = np.linalg.pinv(cov_reg + self.eps_ * np.eye(d))

        return cov_inv

    def fit(self, outputs, labels):
        """Fit the Mahalanobis distance baseline.

        Step 1: Compute per-layer class-conditional means and shared
                covariance from "correct" (label=0) tokens.
        Step 2: Compute Mahalanobis distance per token per layer.
        Step 3: Optionally reduce layer-wise MD features with PCA.
        Step 4: Train Ridge regression to predict uncertainty.

        Args:
            outputs: Dict with 'hidden_states' tensor of shape
                (batch_size, seq_len, n_layers, hidden_size). May also
                contain 'scores' (logits) for sequence log-probability.
            labels: Per-token binary labels (0=factual, 1=hallucinated).
                If answer-level, will be repeated per token.
        """
        hidden_states = outputs['hidden_states']  # (B, seq_len, n_layers, hidden_size)
        n_layers = hidden_states.shape[2]
        hidden_size = hidden_states.shape[3]

        # Flatten to (n_tokens, n_layers, hidden_size)
        B, seq_len = hidden_states.shape[:2]
        flat_embeddings = hidden_states.reshape(-1, n_layers, hidden_size)  # (n_tokens, n_layers, hidden_size)
        n_tokens = flat_embeddings.shape[0]

        # Handle labels
        labels = np.asarray(labels)
        if labels.ndim == 0 or labels.shape == ():
            # Scalar label — repeat for all tokens
            labels = np.full(n_tokens, int(labels), dtype=int)
        elif labels.ndim == 1 and len(labels) == B:
            # Answer-level — repeat per token position
            labels = np.repeat(labels, seq_len)
        elif labels.ndim == 2 and labels.shape == (B, seq_len):
            labels = labels.ravel()
        elif len(labels) != n_tokens:
            # Last resort: try repeating by seq_len
            if len(labels) == B:
                labels = np.repeat(labels, seq_len)
            else:
                raise ValueError(
                    f"Labels shape {labels.shape} incompatible with "
                    f"{n_tokens} tokens (B={B}, seq_len={seq_len})"
                )

        # Split into correct (label=0) and hallucinated (label=1)
        correct_mask = labels == 0
        correct_embeddings = flat_embeddings[correct_mask]  # (n_correct, n_layers, hidden_size)
        n_correct = correct_embeddings.shape[0]

        if n_correct == 0:
            raise ValueError(
                "No 'correct' tokens (label=0) found. "
                "Cannot estimate density of factual distribution."
            )

        # Step 1: Per-layer class-conditional means and shared covariance
        self.class_means_ = []
        cov_inv_list = []
        for layer_idx in range(n_layers):
            layer_emb = correct_embeddings[:, layer_idx, :]  # (n_correct, hidden_size)
            mean_vec = layer_emb.mean(axis=0)
            self.class_means_.append(mean_vec)
            cov_inv = self._estimate_covariance(layer_emb)
            cov_inv_list.append(cov_inv)

        self.shared_cov_inv_ = cov_inv_list

        # Step 2: Compute per-layer MD for ALL tokens
        md_features = self._compute_mahalanobis_per_layer(
            flat_embeddings, self.class_means_, cov_inv_list
        )  # (n_tokens, n_layers)

        # Handle NaN/inf
        if self.handle_nan:
            md_features = self._safe_replace(md_features, fill_value=0.0)

        # Optionally add per-token log-probability as an extra feature
        if self.use_logprob and 'scores' in outputs:
            scores = outputs['scores']  # (B, seq_len, vocab)
            flat_scores = scores.reshape(-1, scores.shape[-1])  # (n_tokens, vocab)
            log_probs = np.log(np.exp(flat_scores).sum(axis=-1) + 1e-10)
            log_probs = log_probs.reshape(-1, 1)
            if self.handle_nan:
                log_probs = self._safe_replace(log_probs, fill_value=0.0)
            md_features = np.concatenate([md_features, log_probs], axis=1)

        # Step 3: PCA on layer-wise (or augmented) features
        n_components = self.n_components
        if n_components is None:
            n_components = min(n_layers + (1 if self.use_logprob and 'scores' in outputs else 0), 10)
        n_components = min(n_components, md_features.shape[1])
        n_components = max(n_components, 1)

        self.pca = PCA(n_components=n_components)
        md_pca = self.pca.fit_transform(md_features)  # (n_tokens, n_components)

        # Scale for regression
        self.scaler = StandardScaler()
        md_scaled = self.scaler.fit_transform(md_pca)

        # Step 4: Train Ridge regression
        self.regressor = Ridge(alpha=self.alpha)
        self.regressor.fit(md_scaled, labels)

        self.feature_dim_ = md_features.shape[1]
        print(
            f"  [TokenMahalanobis] fit complete: {n_layers} layers, "
            f"{n_components} PCA components, {n_correct} correct tokens"
        )

        # Find optimal threshold on training data
        preds = self.regressor.predict(md_scaled)
        from moeuncert.experiments.utils import optimal_threshold
        best_thresh, best_f1 = optimal_threshold(labels, preds)
        self.threshold_ = best_thresh
        print(f"  [TokenMahalanobis] optimal threshold: {best_thresh:.4f} (F1: {best_f1:.4f})")

    def _extract_features(self, outputs):
        """Extract and transform MD features for prediction."""
        hidden_states = outputs['hidden_states']  # (B, seq_len, n_layers, hidden_size)
        n_layers = hidden_states.shape[2]

        B, seq_len = hidden_states.shape[:2]
        flat_embeddings = hidden_states.reshape(-1, n_layers, hidden_states.shape[3])

        # Compute MD per layer
        md_features = self._compute_mahalanobis_per_layer(
            flat_embeddings, self.class_means_, self.shared_cov_inv_
        )
        if self.handle_nan:
            md_features = self._safe_replace(md_features, fill_value=0.0)

        # Optionally add log-prob
        if self.use_logprob and 'scores' in outputs:
            scores = outputs['scores']
            flat_scores = scores.reshape(-1, scores.shape[-1])
            log_probs = np.log(np.exp(flat_scores).sum(axis=-1) + 1e-10).reshape(-1, 1)
            if self.handle_nan:
                log_probs = self._safe_replace(log_probs, fill_value=0.0)
            md_features = np.concatenate([md_features, log_probs], axis=1)

        # PCA + scaling
        md_pca = self.pca.transform(md_features)
        md_scaled = self.scaler.transform(md_pca)

        return md_scaled, B, seq_len

    def predict_proba(self, outputs):
        """Predict continuous uncertainty scores.

        Higher scores = more uncertain / likely hallucinated.

        Args:
            outputs: Dict with 'hidden_states' tensor.
                May also contain 'scores' for log-prob features.

        Returns:
            Uncertainty scores as numpy array.
            If outputs contains a single sample, returns a scalar.
            Otherwise returns per-token array matching flat tokens.
        """
        if self.regressor is None:
            raise RuntimeError("TokenMahalanobis not fitted yet. Call fit() first.")

        md_scaled, B, seq_len = self._extract_features(outputs)
        scores = self.regressor.predict(md_scaled)  # (n_tokens,)

        if len(scores) == 1:
            return float(scores[0])

        # Reshape back to (B, seq_len) for per-token output
        return scores.reshape(B, seq_len)

    def predict(self, outputs):
        """Predict binary hallucination labels.

        Args:
            outputs: Dict with 'hidden_states' tensor.

        Returns:
            Binary labels (0=factual, 1=hallucinated).
            Shape matches predict_proba output.
        """
        scores = self.predict_proba(outputs)
        if isinstance(scores, float):
            return int(scores >= self.threshold_)
        return (scores >= self.threshold_).astype(int)
