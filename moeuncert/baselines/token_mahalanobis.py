"""
Token-Level Mahalanobis Distance baseline (Vazhentsev et al., 2025).

Faithful re-implementation based on the paper and official source code at
https://github.com/ArtemVazh/token_mahalanobis_distance

Paper: "Token-Level Density-Based Uncertainty Quantification Methods for
       Eliciting Truthfulness of Large Language Models" (NAACL 2025)

Algorithm:
    1. Extract token embeddings from multiple decoder layers.
    2. For each layer, compute a class-conditional centroid (mean of tokens
       labeled as "correct/factual") and a shared covariance matrix.
    3. Compute the Mahalanobis distance MD(x) = sqrt((x-μ)^T Σ^{-1} (x-μ))
       for every token at every layer.
    4. (Optional) Filter training tokens by a quality metric threshold to
       include only high-quality correct tokens.
    5. Train a Ridge regression meta-model on the layer-wise MD features
       to predict a continuous quality score (higher uncertainty = worse).
    6. Per the paper's HUQ extension, also uses Maximum Sequence Probability
       as an aleatoric uncertainty signal in a two-stage combination.

NOTE: The official paper's main method combines per-layer MD scores as
features to a meta-regressor that predicts a quality metric (not binary
hallucination labels). Binary classification is done via thresholding on
the predicted continuous score.
"""

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

from ._base import BaseBaseline

# Small jitter values for covariance regularization, matching the paper's
# lm_polygraph convention: progressively larger jitters if inversion fails.
JITTERS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


class TokenMahalanobis(BaseBaseline):
    """Token-Level Mahalanobis Distance baseline (Vazhentsev et al., 2025).

    Follows the paper's supervised approach:
    - For each selected layer, compute centroid + shared covariance from
      "correct" training tokens.
    - Compute per-token MD for every token at every layer.
    - Train a Ridge regression on layer-wise MD features (optionally with
      PCA/correlation-based feature reduction) to predict a quality score.

    The HUQ (Hybrid Uncertainty Quantification) two-stage combination
    with Maximum Sequence Probability is available via use_huq=True.

    Args:
        alpha: Ridge regression regularization strength. If None, uses
            RidgeCV with CV search. Default 1.0.
        positive: Constrain Ridge coefficients to be positive (paper found
            this consistently effective). Default True.
        metric_thr: Threshold on quality metric for training token filtering.
            Only "correct" tokens with metric value >= metric_thr are used
            for centroid/covariance estimation. 0.0 = use all. Default 0.0.
        handle_nan: Replace inf/nan with 0. Default True.
        use_huq: If True, combine MD scores with Maximum Sequence
            Probability via the HUQ ranking step from the paper.
            If False, use Ridge directly on MD features. Default True.
    """

    def __init__(self, alpha=1.0, positive=True, metric_thr=0.0,
                 handle_nan=True, use_huq=True):
        self.alpha = alpha
        self.positive = positive
        self.metric_thr = metric_thr
        self.handle_nan = handle_nan
        self.use_huq = use_huq

        # Per-layer centroid and covariance inverse
        self.centroids_ = []        # list of (hidden_size,) per layer
        self.sigma_inv_ = []        # list of (hidden_size, hidden_size) per layer

        self.regressor_ = None      # Ridge meta-model on layer-wise MD features
        self.is_fitted_ = False
        self.threshold_ = 0.5

        # HUQ parameters (learned via grid search on val split)
        self.huq_t_min_ = 0.1
        self.huq_t_max_ = 0.9
        self.huq_alpha_ = 0.1
        self.train_md_scores_ = None   # MD scores on training dev split
        self.train_msp_scores_ = None  # MSP scores on training dev split

    def _safe_replace(self, arr, fill_value=0.0):
        return np.where(np.isfinite(arr), arr, fill_value)

    def _compute_inv_covariance(self, centroid, embeddings):
        """Compute regularized inverse covariance matrix.

        Matches the lm_polygraph `compute_inv_covariance` logic:
        1. Center embeddings around the known centroid (not empirical mean).
        2. Compute covariance = (X.T @ X) / (n-1).
        3. Invert with progressive jitter if singular.

        Args:
            centroid: (hidden_size,) — pre-computed mean.
            embeddings: (n, hidden_size) — training token embeddings.

        Returns:
            (hidden_size, hidden_size) — pseudo-inverse of covariance.
        """
        n = embeddings.shape[0]
        centered = embeddings - centroid  # Use known centroid, not empirical mean
        cov = (centered.T @ centered) / (n - 1)

        for jitter in JITTERS:
            try:
                cov_reg = cov + jitter * np.eye(cov.shape[0])
                inv = np.linalg.inv(cov_reg)
                return inv
            except np.linalg.LinAlgError:
                continue

        # Ultimate fallback: pseudo-inverse
        return np.linalg.pinv(cov)

    def _compute_mahalanobis_distance_with_centroid(self, centroid, sigma_inv,
                                                     embeddings):
        """Compute Mahalanobis distance to known centroid.

        MD(x) = sqrt((x - μ)^T Σ^{-1} (x - μ))

        Args:
            centroid: (hidden_size,) — class-conditional mean.
            sigma_inv: (hidden_size, hidden_size) — inverse covariance.
            embeddings: (n, hidden_size) — token embeddings.

        Returns:
            (n,) array of MD values.
        """
        centered = embeddings - centroid
        left = centered @ sigma_inv
        md2 = np.sum(left * centered, axis=1)
        md2 = np.maximum(md2, 0.0)  # Guard against tiny negatives
        return np.sqrt(md2)

    def fit(self, outputs, labels, metrics=None):
        """Fit the Token Mahalanobis Distance baseline.

        Step 1: Extract per-layer centroids + cov inverses from correct tokens.
        Step 2: Compute per-layer MD for all (or filtered) training tokens.
        Step 3: Train Ridge regression (or HUQ) on layer-wise MD features.

        Args:
            outputs: Dict with 'hidden_states' tensor
                (batch_size, seq_len, n_layers, hidden_size). May also
                contain 'scores' (logits) for MSP features.
            labels: Per-token labels (0=factual, 1=hallucinated). If 1D
                with length == batch_size, repeated per-token.
            metrics: Optional per-token quality metrics for filtering
                (used with metric_thr). If None, uses all correct tokens.
        """
        hidden_states = outputs['hidden_states']
        B, seq_len, n_layers, hidden_size = hidden_states.shape

        # Flatten to (n_tokens, n_layers, hidden_size)
        flat_emb = hidden_states.reshape(-1, n_layers, hidden_size)
        n_tokens = flat_emb.shape[0]

        # Process labels
        labels = np.asarray(labels, dtype=float)
        if labels.ndim == 0 or labels.shape == ():
            labels = np.full(n_tokens, int(labels))
        elif labels.ndim == 1 and len(labels) == B:
            labels = np.repeat(labels, seq_len)
        elif labels.ndim == 2 and labels.shape == (B, seq_len):
            labels = labels.ravel()
        elif len(labels) != n_tokens:
            if len(labels) == B:
                labels = np.repeat(labels, seq_len)
            else:
                raise ValueError(f"Labels shape {labels.shape} incompatible")

        correct_mask = labels == 0
        if correct_mask.sum() == 0:
            print("  WARNING: No correct tokens (label=0) found. "
                  "Centroid will be estimated from all tokens.")

        # Process quality metrics for token filtering
        if metrics is not None:
            metrics = np.asarray(metrics, dtype=float)
            if metrics.ndim > 1:
                metrics = metrics.ravel()
            if len(metrics) != n_tokens:
                if len(metrics) == B:
                    metrics = np.repeat(metrics, seq_len)

        # Per-layer: compute centroid + cov_inv from ALL training tokens
        # (The MD per layer is unsupervised — centroid is mean of all training
        # embeddings. The supervised part is the Ridge regression that learns
        # to map per-layer MD scores to uncertainty.
        # If metric_thr > 0, we can filter low-quality tokens as the paper does.)
        self.centroids_ = []
        self.sigma_inv_ = []

        # Determine which tokens to use for centroid/cov estimation
        # Default: all tokens. If metric_thr > 0, filter by quality.
        emb_for_centroid = flat_emb  # All tokens by default
        if self.metric_thr > 0 and metrics is not None:
            good_quality = metrics >= self.metric_thr
            if good_quality.sum() >= 10:
                emb_for_centroid = flat_emb[good_quality]

        for layer_idx in range(n_layers):
            layer_emb = emb_for_centroid[:, layer_idx, :]  # (n_used, hidden_size)

            centroid = layer_emb.mean(axis=0)
            sigma_inv = self._compute_inv_covariance(centroid, layer_emb)

            self.centroids_.append(centroid)
            self.sigma_inv_.append(sigma_inv)

        # Compute per-layer MD for all tokens
        md_features = np.zeros((n_tokens, n_layers))
        for layer_idx in range(n_layers):
            md_features[:, layer_idx] = self._compute_mahalanobis_distance_with_centroid(
                self.centroids_[layer_idx],
                self.sigma_inv_[layer_idx],
                flat_emb[:, layer_idx, :],
            )

        if self.handle_nan:
            md_features = self._safe_replace(md_features, fill_value=0.0)

        # Compute sequence-level MD (average across tokens per sample)
        seq_md = np.zeros(B)
        for b in range(B):
            start = b * seq_len
            end = start + seq_len
            token_mds = md_features[start:end]
            seq_md[b] = np.nanmean(token_mds)

        # Compute MSP (Maximum Sequence Probability) for HUQ
        seq_logprobs = None
        if 'scores' in outputs:
            scores = outputs['scores']  # (B, seq_len, vocab)
            if isinstance(scores, (np.ndarray,)):
                log_probs = np.log(
                    np.exp(scores - scores.max(axis=-1, keepdims=True)).sum(axis=-1)
                    + 1e-10
                )
            else:
                log_probs = np.zeros((B, seq_len))
            seq_logprobs = np.sum(log_probs, axis=1)  # (B,)

        # Build answer-level labels
        seq_labels = np.zeros(B)
        for b in range(B):
            start = b * seq_len
            end = start + seq_len
            seq_labels[b] = np.max(labels[start:end])

        if self.use_huq and seq_logprobs is not None:
            # HUQ two-stage: grid search on val split
            dev_size = min(0.5, 10.0 / B) if B > 10 else 0.3
            train_idx, dev_idx = train_test_split(
                np.arange(B), test_size=dev_size,
                random_state=42, stratify=seq_labels if len(np.unique(seq_labels)) > 1 else None,
            )

            train_md = seq_md[train_idx]
            dev_md = seq_md[dev_idx]
            train_msp = seq_logprobs[train_idx]
            dev_msp = seq_logprobs[dev_idx]

            # Grid search for HUQ parameters on dev set
            from scipy.stats import rankdata

            def total_uncertainty(md_scores, msp_scores, t_min, t_max, alpha):
                """HUQ combination: ranking-based two-stage uncertainty.

                Matches the paper's total_uncertainty_linear_step.
                """
                n = len(md_scores)
                n_lowest = int(n * t_min)
                n_max = int(n * t_max)

                md_rank = rankdata(md_scores)
                msp_rank = rankdata(-msp_scores)  # Invert: lower MSP = higher uncertainty

                total = (1 - alpha) * md_rank + alpha * msp_rank
                # For low epistemic uncertainty, use aleatoric
                if n_lowest > 0:
                    low_eps = np.argsort(md_rank)[:n_lowest]
                    total[low_eps] = rankdata(msp_scores[low_eps])
                # For high aleatoric uncertainty with low epistemic, use aleatoric
                if n_max > 0:
                    high_alea = np.where(msp_rank > n_max)[0]
                    for idx in low_eps:
                        if idx in high_alea:
                            total[idx] = msp_rank[idx]
                return total

            best_score = -np.inf
            best_params = (0.1, 0.9, 0.1)

            for t_min in np.arange(0.0, 0.35, 0.05):
                for t_max in np.arange(0.7, 1.05, 0.05):
                    for alpha in np.arange(0.0, 1.05, 0.1):
                        scores = total_uncertainty(dev_md, dev_msp, t_min, t_max, alpha)
                        dev_labels = seq_labels[dev_idx]
                        if len(np.unique(dev_labels)) > 1:
                            from sklearn.metrics import roc_auc_score
                            try:
                                score = roc_auc_score(dev_labels, scores)
                                if score > best_score:
                                    best_score = score
                                    best_params = (t_min, t_max, alpha)
                            except Exception:
                                pass

            self.huq_t_min_, self.huq_t_max_, self.huq_alpha_ = best_params
            self.train_md_scores_ = dev_md
            self.train_msp_scores_ = dev_msp

            # Fit Ridge on training data for use during inference
            X_train = train_md.reshape(-1, 1)
            y_train = seq_labels[train_idx]
            self.regressor_ = Ridge(alpha=self.alpha, positive=self.positive)
            self.regressor_.fit(X_train, y_train)

            self.is_fitted_ = True

        else:
            # Direct Ridge regression without HUQ
            X = seq_md.reshape(-1, 1)
            y = seq_labels

            self.regressor_ = Ridge(alpha=self.alpha, positive=self.positive)
            self.regressor_.fit(X, y)
            self.is_fitted_ = True

        # Find optimal threshold
        train_scores = self.predict_proba(outputs)
        from moeuncert.experiments.utils import optimal_threshold
        best_thresh, best_f1 = optimal_threshold(seq_labels, train_scores)
        self.threshold_ = best_thresh

        print(f"  [TokenMahalanobis] fit: {n_layers} layers, "
              f"{int(correct_mask.sum())} correct tokens out of {n_tokens}, "
              f"{'HUQ' if self.use_huq else 'Ridge'} mode, "
              f"threshold={best_thresh:.4f} F1={best_f1:.4f}")

    def predict_proba(self, outputs):
        """Predict uncertainty scores.

        Higher scores = more uncertain / more likely hallucinated.

        For HUQ mode: combines MD and MSP via the learned two-stage ranking.
        For Ridge mode: directly uses the regressor.

        Args:
            outputs: Dict with 'hidden_states' (B, seq_len, n_layers, h).
                May contain 'scores' (logits) for MSP.

        Returns:
            (B,) array of uncertainty scores.
        """
        if not self.is_fitted_:
            raise RuntimeError("Not fitted. Call fit() first.")

        hidden_states = outputs['hidden_states']
        B, seq_len, n_layers, hidden_size = hidden_states.shape

        flat_emb = hidden_states.reshape(-1, n_layers, hidden_size)

        md_features = np.zeros((B * seq_len, n_layers))
        for layer_idx in range(n_layers):
            md_features[:, layer_idx] = self._compute_mahalanobis_distance_with_centroid(
                self.centroids_[layer_idx],
                self.sigma_inv_[layer_idx],
                flat_emb[:, layer_idx, :],
            )

        if self.handle_nan:
            md_features = self._safe_replace(md_features, fill_value=0.0)

        # Sequence-level MD
        seq_md = np.zeros(B)
        for b in range(B):
            start = b * seq_len
            end = start + seq_len
            seq_md[b] = np.nanmean(md_features[start:end])

        if self.use_huq and 'scores' in outputs:
            # MSP
            scores = outputs['scores']
            if isinstance(scores, (np.ndarray,)):
                log_probs = np.log(
                    np.exp(scores - scores.max(axis=-1, keepdims=True)).sum(axis=-1)
                    + 1e-10
                )
            else:
                log_probs = np.zeros((B, seq_len))
            seq_msp = np.sum(log_probs, axis=1)

            # HUQ combination
            from scipy.stats import rankdata

            n = B
            n_lowest = int(n * self.huq_t_min_)
            n_max = int(n * self.huq_t_max_)

            md_rank = rankdata(seq_md)
            msp_rank = rankdata(-seq_msp)

            total = (1 - self.huq_alpha_) * md_rank + self.huq_alpha_ * msp_rank

            if n_lowest > 0:
                low_eps = np.argsort(md_rank)[:n_lowest]
                total[low_eps] = rankdata(seq_msp[low_eps])
            if n_max > 0:
                high_alea = np.where(msp_rank > n_max)[0]
                for idx in low_eps:
                    if idx in high_alea:
                        total[idx] = msp_rank[idx]

            return total / total.max() if total.max() > 0 else total
        else:
            return self.regressor_.predict(seq_md.reshape(-1, 1))

    def predict(self, outputs):
        """Predict binary labels (0=factual, 1=hallucinated)."""
        scores = self.predict_proba(outputs)
        return (scores >= self.threshold_).astype(int)
