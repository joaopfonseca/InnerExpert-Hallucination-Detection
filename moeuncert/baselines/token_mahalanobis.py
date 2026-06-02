"""
Token-Level Mahalanobis Distance baseline (Vazhentsev et al., 2025).

Faithful re-implementation based on the paper and official source code at
https://github.com/ArtemVazh/token_mahalanobis_distance

Paper: "Token-Level Density-Based Uncertainty Quantification Methods for
       Eliciting Truthfulness of Large Language Models" (NAACL 2025)

Algorithm (following the official codebase):

    === TokenMahalanobisDistance (per-layer unsupervised MD) ===
    1. For each layer (or one layer), extract token embeddings.
    2. Compute centroid = mean embedding of training tokens (optionally
       filtered by quality metric threshold).
    3. Compute regularized inverse covariance from training embeddings
       centered around the known centroid.
    4. MD(x) = sqrt((x - μ)^T Σ^{-1} (x - μ)) for every token.
    5. Aggregate per-sequence: mean or sum across tokens.

    === LinRegTokenMahalanobisDistance (supervised meta-model) ===
    6. Split training data: 50% train centroids, 50% dev MD + meta-model.
    7. For each hidden layer, compute per-layer MD and aggregate to
       sequence-level scores.
    8. Train Ridge regression on multi-layer MD features (with optional
       rankdata normalization) to predict a continuous quality metric.
    9. Use the Ridge predictions (= y_preds) as the epistemic signal.

    === HUQ_LRTMD (Hybrid Uncertainty Quantification) ===
    10. Use MD-based y_preds as epistemic, MSP as aleatoric.
    11. Grid-search HUQ parameters (t_min, t_max, alpha) maximizing
        Prediction Rejection Area (PRR) on the dev set.
    12. At inference: concatenate train+test, compute total_uncertainty,
        then strip training portion.

NOTE: This simplified version does per-sequence aggregation (mean MD
across tokens per answer) as the base MD feature, then follows the
same Ridge → HUQ pipeline. The full official code supports per-layer
MD as multi-dimensional features with optional decorrelation/PCA.
"""

import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import rankdata

from ._base import BaseBaseline

# Jitter values for covariance regularization, matching lm_polygraph
# JITTERS = [10**exp for exp in range(-15, 0, 1)]
# = [1e-15, 1e-14, ..., 1e-1] (15 values, NO 1.0).
# Loop breaks at first PSD-passing jitter (eigenvalue check).
JITTERS = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
           1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
           1e-3, 1e-2, 1e-1]


class TokenMahalanobis(BaseBaseline):
    """Token-Level Mahalanobis Distance baseline (Vazhentsev et al., 2025).

    Models the official LinRegTokenMahalanobisDistance + HUQ_LRTMD pipeline.

    Step 1: Per-layer centroid + cov_inv from ALL training tokens.
    Step 2: Per-layer MD → sequence-level aggregation → Ridge regression.
    Step 3: (If use_huq) HUQ combination of Ridge predictions (epistemic)
            + MSP (aleatoric) via ranking-based two-stage formula.

    Args:
        alpha: Ridge regularization strength. Default 1.0.
        positive: Constrain Ridge coefficients to be positive. Default True.
        metric_thr: Quality metric threshold for token filtering when
            estimating centroid/covariance. 0 = no filtering. Default 0.0.
        handle_nan: Replace inf/nan with 0. Default True.
        use_huq: If True, use HUQ (MD + MSP ranking combination).
            If False, use raw Ridge predictions. Default True.
    """

    def __init__(self, alpha=1.0, positive=True, metric_thr=0.0,
                 handle_nan=True, use_huq=True):
        self.alpha = alpha
        self.positive = positive
        self.metric_thr = metric_thr
        self.handle_nan = handle_nan
        self.use_huq = use_huq

        # Per-layer centroid and covariance inverse
        self.centroids_ = []
        self.sigma_inv_ = []

        self.regressor_ = None
        self.is_fitted_ = False
        self.threshold_ = 0.5

        # HUQ parameters (learned via grid search on dev split)
        self.huq_t_min_ = 0.1
        self.huq_t_max_ = 0.9
        self.huq_alpha_ = 0.1
        self.train_dev_md_ = None    # Ridge predictions on dev split
        self.train_dev_msp_ = None   # MSP on dev split

    def _safe_replace(self, arr, fill_value=0.0):
        return np.where(np.isfinite(arr), arr, fill_value)

    def _compute_inv_covariance(self, centroid, embeddings):
        """Compute regularized inverse covariance matrix.

        Matches lm_polygraph `compute_inv_covariance`:
        1. Cov = (X - μ)^T (X - μ) / (n - 1) where μ = embeddings.mean(0).
           (When centroid == mean(embeddings), this is equivalent to cov).
        2. Add progressive jitter, check positive semidefinite via
           eigenvalue check (matching official), then invert.

        Args:
            centroid: (hidden_size,) — pre-computed mean.
            embeddings: (n, hidden_size) — training token embeddings.

        Returns:
            (hidden_size, hidden_size) inverse covariance.
        """
        n = embeddings.shape[0]
        centered = embeddings - centroid
        cov = (centered.T @ centered) / (n - 1)
        # Match official torch.nan_to_num on the covariance
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

        for jitter in JITTERS:
            cov_reg = cov + jitter * np.eye(cov.shape[0])
            try:
                eigenvalues = np.linalg.eigh(cov_reg)[0]
                if (eigenvalues >= 0).all():
                    return np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                continue

        # Ultimate fallback
        return np.linalg.pinv(cov)

    def _compute_mahalanobis_distance(self, centroid, sigma_inv, embeddings):
        """MD(x) = sqrt((x - μ)^T Σ^{-1} (x - μ))

        Matches mahalanobis_distance_with_known_centroids_sigma_inv.
        """
        centered = embeddings - centroid
        left = centered @ sigma_inv
        md2 = np.maximum(np.sum(left * centered, axis=1), 0.0)
        return np.sqrt(md2)

    def fit(self, outputs, labels, metrics=None):
        """Fit the Token Mahalanobis Distance baseline + optional HUQ.

        Args:
            outputs: Dict with 'hidden_states'
                (B, seq_len, n_layers, hidden_size). May also contain
                'scores' (logits) for MSP, and 'greedy_log_likelihoods'
                for direct MSP access.
            labels: Binary labels (0=factual, 1=hallucinated). Can be
                per-token or per-answer.
            metrics: Optional per-token quality metrics for filtering
                (used with metric_thr).
        """
        hidden_states = outputs['hidden_states']
        B, seq_len, n_layers, hidden_size = hidden_states.shape

        flat_emb = hidden_states.reshape(-1, n_layers, hidden_size)
        n_tokens = flat_emb.shape[0]

        # Process labels to per-token
        labels = np.asarray(labels, dtype=float)
        if labels.ndim == 0:
            labels = np.full(n_tokens, int(labels))
        elif labels.ndim == 1 and len(labels) == B:
            labels = np.repeat(labels, seq_len)
        elif labels.ndim == 2 and labels.shape == (B, seq_len):
            labels = labels.ravel()
        elif len(labels) != n_tokens and len(labels) == B:
            labels = np.repeat(labels, seq_len)

        # Per-answer labels (for later threshold finding)
        seq_labels = np.max(labels.reshape(B, seq_len), axis=1)

        # ------------------------
        # Step 1: Split train/dev before centroid computation
        #   The official LinRegTokenMahalanobisDistance splits 50/50 first
        #   (random_state=42) — non-stratified — and computes centroids ONLY
        #   from train tokens. This avoids data leakage from dev tokens into
        #   the centroid.
        # ------------------------
        from sklearn.model_selection import train_test_split
        train_idx, dev_idx = train_test_split(
            list(range(B)), test_size=0.5, shuffle=True, random_state=42,
        )

        # Map token indices for training samples
        token_train_idx = np.concatenate([
            np.arange(b * seq_len, (b + 1) * seq_len)
            for b in train_idx
        ])
        train_flat_emb = flat_emb[token_train_idx]

        # ------------------------
        # Step 2: Per-layer centroid + cov_inv from TRAINING tokens only
        #   Following the official TokenMahalanobisDistance:
        #   centroid = mean of train embeddings (filtered by metric_thr)
        # ------------------------
        emb_for_centroid = train_flat_emb
        if self.metric_thr > 0 and metrics is not None:
            metrics_arr = np.asarray(metrics, dtype=float).ravel()
            if len(metrics_arr) != n_tokens:
                if len(metrics_arr) == B:
                    metrics_arr = np.repeat(metrics_arr, seq_len)
            # Only consider training tokens for filtering
            train_metrics = metrics_arr[token_train_idx]
            good_train = train_metrics >= self.metric_thr
            if good_train.sum() >= 10:
                emb_for_centroid = train_flat_emb[good_train]

        self.centroids_ = []
        self.sigma_inv_ = []

        for layer_idx in range(n_layers):
            layer_emb = emb_for_centroid[:, layer_idx, :]
            centroid = layer_emb.mean(axis=0)
            sigma_inv = self._compute_inv_covariance(centroid, layer_emb)
            self.centroids_.append(centroid)
            self.sigma_inv_.append(sigma_inv)

        # ------------------------
        # Step 3: Per-layer MD for ALL tokens, then sequence-level
        # aggregation PER LAYER (matching official LinRegTokenMahalanobisDistance)
        # Official: for each layer, compute MD scores, then average across tokens
        # per sequence -> train_dists has shape (dev_samples, n_layers)
        # ------------------------
        md_features = np.zeros((n_tokens, n_layers))
        for layer_idx in range(n_layers):
            md_features[:, layer_idx] = self._compute_mahalanobis_distance(
                self.centroids_[layer_idx],
                self.sigma_inv_[layer_idx],
                flat_emb[:, layer_idx, :],
            )

        if self.handle_nan:
            md_features = self._safe_replace(md_features, fill_value=0.0)

        # Sequence-level MD: per-layer average across tokens -> (B, n_layers)
        seq_md = np.array([
            md_features[b*seq_len:(b+1)*seq_len].mean(axis=0)
            for b in range(B)
        ])  # Shape (B, n_layers)

        # rankdata normalization (matching norm="norm" in official code)
        # seq_md shape: (B, n_layers)
        X_raw = seq_md.copy()
        X_norm = np.zeros_like(X_raw)
        for col in range(X_norm.shape[1]):
            X_norm[:, col] = rankdata(X_raw[:, col])
            X_norm[:, col] /= X_norm[:, col].max()

        # ------------------------
        # Step 3.5: Fit Ridge on DEV split and predict on DEV (in-sample)
        #   Matching official LinRegTokenMahalanobisDistance exactly:
        #     self.regressor.fit(X, y)            # X is dev features
        #     self.y_preds = self.regressor.predict(X)  # in-sample dev preds
        #   The official's y = 1 - target (continuous quality metric).
        #   Our adaptation uses binary labels, so target = seq_labels and
        #   y = 1 - target = 1 - seq_labels (so factual=1, hallucinated=0,
        #   matching the "higher = better" convention).
        #   These in-sample dev predictions become train_md for HUQ.
        # ------------------------
        X_dev = X_norm[dev_idx]
        # 1 - seq_labels gives "quality" (factual=1, hallucinated=0),
        # matching the official `1 - target` convention.
        y_dev = 1.0 - seq_labels[dev_idx]
        # np.nan_to_num equivalent: if any quality is NaN, replace with 0
        y_dev = np.nan_to_num(y_dev, nan=0.0)
        self.regressor_ = Ridge(alpha=self.alpha, positive=self.positive)
        self.regressor_.fit(X_dev, y_dev)
        # Predict on the same dev split (in-sample, like official)
        dev_preds = self.regressor_.predict(X_dev)

        # ------------------------
        # Step 4: HUQ optional (following HUQ_LRTMD)
        # ------------------------
        if self.use_huq and 'greedy_log_likelihoods' in outputs:
            gll = outputs['greedy_log_likelihoods']  # (B, seq_len) or (B,)
            gll = np.asarray(gll, dtype=float)
            if gll.ndim > 1:
                # Official MaximumSequenceProbability: -sum(log_likelihoods)
                # Higher = more uncertain. Matches lm_polygraph source.
                seq_msp = -np.sum(gll, axis=1)
            else:
                seq_msp = -np.asarray(gll)

            dev_msp = seq_msp[dev_idx]

            def total_uncertainty(epistemic, aleatoric, t_min, t_max, alpha):
                """Matching official total_uncertainty_linear_step exactly."""
                n = len(epistemic)
                n_lowest = int(n * t_min)
                n_max = int(n * t_max)

                alea_rank = rankdata(aleatoric)
                epi_rank = rankdata(epistemic)

                total = (1 - alpha) * epi_rank + alpha * alea_rank
                total[epi_rank <= n_lowest] = rankdata(aleatoric[epi_rank <= n_lowest])
                total[(alea_rank > n_max) & (epi_rank <= n_lowest)] = \
                    alea_rank[(alea_rank > n_max) & (epi_rank <= n_lowest)]
                return total

            # Official HUQ_LRTMD uses dev set directly (no duplication)
            combined_epi = dev_preds
            combined_alea = dev_msp
            # Labels: convert to quality metric where higher = better
            # (1 - label) so factual (0) → 1.0, hallucinated (1) → 0.0
            combined_quality = 1.0 - seq_labels[dev_idx]

            # Official best_prr starts with PRR of raw epistemic signal
            from sklearn.metrics import roc_auc_score

            def compute_prr_proxy(unc_scores, quality):
                """PRR proxy: AUROC of uncertainty scores vs quality.
                Higher unc for positive class (hallucination=higher uncertainty)
                means higher AUROC = better detection.
                This approximates PRR directionally."""
                if len(np.unique(quality)) > 1:
                    return roc_auc_score(quality, unc_scores)
                return -np.inf

            best_prr = compute_prr_proxy(combined_epi, combined_quality)
            for t_min in np.arange(0.0, 0.31, 0.05):
                for t_max in np.arange(0.8, 1.01, 0.05):
                    for alpha in np.arange(0.0, 1.01, 0.1):
                        unc = total_uncertainty(combined_epi, combined_alea,
                                                 t_min, t_max, alpha)
                        score = compute_prr_proxy(unc, combined_quality)
                        if score > best_prr:
                            best_prr = score
                            self.huq_t_min_ = t_min
                            self.huq_t_max_ = t_max
                            self.huq_alpha_ = alpha

            # Store train dev predictions for inference-time concatenation
            self.train_dev_md_ = dev_preds
            self.train_dev_msp_ = dev_msp

        else:
            # Fallback: MSP via scores (logits)
            if self.use_huq and 'scores' in outputs:
                scores = outputs['scores']
                if isinstance(scores, np.ndarray):
                    logits = scores
                    if logits.ndim == 3:
                        logits = logits[:, -1, :]
                    logsum = np.log(
                        np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1)
                        + 1e-10
                    )
                    log_probs = logits - logsum[:, None]
                    # Negative log max probability per token, sum across sequence
                    seq_msp = -np.sum(np.max(log_probs, axis=-1), axis=1)
                else:
                    seq_msp = np.zeros(B)

                def total_uncertainty(epistemic, aleatoric, t_min, t_max, alpha):
                    n = len(epistemic)
                    n_lowest = int(n * t_min)
                    n_max = int(n * t_max)
                    alea_rank = rankdata(aleatoric)
                    epi_rank = rankdata(epistemic)
                    total = (1 - alpha) * epi_rank + alpha * alea_rank
                    total[epi_rank <= n_lowest] = rankdata(aleatoric[epi_rank <= n_lowest])
                    total[(alea_rank > n_max) & (epi_rank <= n_lowest)] = \
                        alea_rank[(alea_rank > n_max) & (epi_rank <= n_lowest)]
                    return total

                dev_msp = seq_msp[dev_idx]
                combined_epi = dev_preds
                combined_alea = dev_msp
                combined_quality = 1.0 - seq_labels[dev_idx]

                from sklearn.metrics import roc_auc_score

                def compute_prr_proxy(unc_scores, quality):
                    if len(np.unique(quality)) > 1:
                        return roc_auc_score(quality, unc_scores)
                    return -np.inf

                best_prr = compute_prr_proxy(combined_epi, combined_quality)
                for t_min in np.arange(0.0, 0.31, 0.05):
                    for t_max in np.arange(0.8, 1.01, 0.05):
                        for alpha in np.arange(0.0, 1.01, 0.1):
                            unc = total_uncertainty(combined_epi, combined_alea,
                                                     t_min, t_max, alpha)
                            score = compute_prr_proxy(unc, combined_quality)
                            if score > best_prr:
                                best_prr = score
                                self.huq_t_min_ = t_min
                                self.huq_t_max_ = t_max
                                self.huq_alpha_ = alpha
                self.train_dev_md_ = dev_preds
                self.train_dev_msp_ = dev_msp

        self.is_fitted_ = True

        # Find optimal threshold
        train_scores = self.predict_proba(outputs)
        from moeuncert.experiments.utils import optimal_threshold
        best_thresh, best_f1 = optimal_threshold(seq_labels, train_scores)
        self.threshold_ = best_thresh

        print(f"  [TokenMahalanobis] fit: {n_layers} layers, "
              f"B={B}, {'HUQ' if self.use_huq else 'Ridge'} mode, "
              f"threshold={best_thresh:.4f} F1={best_f1:.4f}")

    def predict_proba(self, outputs):
        """Predict uncertainty scores.

        For HUQ: concatenate train dev + test predictions, rank together
        via total_uncertainty_linear_step, then strip training portion.
        For Ridge only: directly return Ridge predictions.

        Args:
            outputs: Dict with 'hidden_states' (B, seq_len, n_layers, h).
                May contain 'greedy_log_likelihoods' or 'scores' for MSP.

        Returns:
            (B,) array of uncertainty scores (higher = more uncertain).
        """
        if not self.is_fitted_:
            raise RuntimeError("Not fitted. Call fit() first.")

        hidden_states = outputs['hidden_states']
        B, seq_len, n_layers, hidden_size = hidden_states.shape

        flat_emb = hidden_states.reshape(-1, n_layers, hidden_size)
        n_tokens = flat_emb.shape[0]

        md_features = np.zeros((n_tokens, n_layers))
        for layer_idx in range(n_layers):
            md_features[:, layer_idx] = self._compute_mahalanobis_distance(
                self.centroids_[layer_idx],
                self.sigma_inv_[layer_idx],
                flat_emb[:, layer_idx, :],
            )

        if self.handle_nan:
            md_features = self._safe_replace(md_features, fill_value=0.0)

        # Sequence-level MD per layer → per-column rankdata normalization
        seq_md = np.array([
            md_features[b*seq_len:(b+1)*seq_len].mean(axis=0)
            for b in range(B)
        ])  # Shape (B, n_layers)
        X_norm = np.zeros_like(seq_md)
        for col in range(X_norm.shape[1]):
            X_norm[:, col] = rankdata(seq_md[:, col])
            X_norm[:, col] /= X_norm[:, col].max()

        md_preds = self.regressor_.predict(X_norm)

        if not self.use_huq:
            return md_preds

        # HUQ: need MSP (MaximumSequenceProbability = -sum(log_likelihoods))
        seq_logprob = None
        if 'greedy_log_likelihoods' in outputs:
            gll = np.asarray(outputs['greedy_log_likelihoods'], dtype=float)
            if gll.ndim > 1:
                seq_logprob = -np.sum(gll, axis=1)
            else:
                seq_logprob = -np.asarray(gll)
        elif 'scores' in outputs:
            scores = outputs['scores']
            if isinstance(scores, np.ndarray):
                logits = scores
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                logsum = np.log(
                    np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1)
                    + 1e-10
                )
                log_probs = logits - logsum[:, None]
                seq_logprob = -np.sum(np.max(log_probs, axis=-1), axis=1)
            else:
                seq_logprob = np.zeros(B)

        if seq_logprob is None:
            return md_preds

        # Concatenate train dev + test (matching official code)
        msp_all = np.concatenate([self.train_dev_msp_, seq_logprob])
        md_all = np.concatenate([self.train_dev_md_, md_preds])

        def total_uncertainty(epistemic, aleatoric, t_min, t_max, alpha):
            n = len(epistemic)
            n_lowest = int(n * t_min)
            n_max = int(n * t_max)
            alea_rank = rankdata(aleatoric)
            epi_rank = rankdata(epistemic)
            total = (1 - alpha) * epi_rank + alpha * alea_rank
            total[epi_rank <= n_lowest] = rankdata(aleatoric[epi_rank <= n_lowest])
            total[(alea_rank > n_max) & (epi_rank <= n_lowest)] = \
                alea_rank[(alea_rank > n_max) & (epi_rank <= n_lowest)]
            return total

        all_scores = total_uncertainty(
            md_all, msp_all,
            self.huq_t_min_, self.huq_t_max_, self.huq_alpha_,
        )

        # Strip training portion (matching official: ues = ues[:len(msp_eval)])
        return all_scores[len(self.train_dev_msp_):]

    def predict(self, outputs):
        """Predict binary labels (0=factual, 1=hallucinated)."""
        scores = self.predict_proba(outputs)
        return (scores >= self.threshold_).astype(int)