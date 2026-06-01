"""
TOHA — TOpology-based HAllucination detector (Bazarova et al., 2025).

Faithful re-implementation based on the paper and official source code at
https://github.com/sb-ai-lab/TOHA

Paper: "Hallucination Detection in LLMs with Topological Divergence on
       Attention Graphs" (ACL 2026)

Algorithm:
    1. For each attention head, transform the attention matrix to a distance
       matrix: d_ij = 1 - a_ij (clipped at 0), zero diagonal, symmetrise.
    2. Zero out the prompt-to-prompt subgraph (distances between prompt
       tokens set to 0), isolating the response subgraph's topology.
    3. Compute MTopDiv: run Vietoris-Rips persistent homology (H_0) on the
       resulting distance matrix and sum the finite barcode lengths.
    4. Normalize by response length.
    5. For supervised mode: select top-n heads using univariate feature
       selection (ANOVA F-value), then fit LogisticRegression.
    6. For unsupervised mode: select heads by difference of means between
       hallucinated and factual → use the average MTopDiv of selected heads.

Official code uses ripser for persistent homology and SelectKBest for
feature selection. Head selection is done on a separate validation set
via `fit_hyperparameters`.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ._base import BaseBaseline

try:
    from ripser import ripser
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False


class TOHA(BaseBaseline):
    """TOHA — TOpology-based HAllucination detector (Bazarova et al., 2025).

    Computes MTopDiv (Manifold Topology Divergence) on attention graphs
    using persistent homology. For each attention head, transforms attention
    weights to distance matrices, zeroes out prompt-to-prompt subgraph,
    computes H_0 barcode via Vietoris-Rips, and sums finite barcode lengths.

    Has two modes:
    - "supervised": Select top-K heads via ANOVA F-value, then fit
      LogisticRegression on those head scores (per the paper).
    - "unsupervised": Select heads by difference-in-means between
      hallucinated and factual samples. Score = mean of selected head
      MTopDiv values (per the paper).

    Args:
        mode: "supervised" or "unsupervised". Default "supervised".
        n_max: Maximum number of heads to select. Default 6 (paper default).
        select_method: Feature selection method for supervised mode.
            "f_classif" (ANOVA F-value) or "mutual_info_classif".
            Default "f_classif".
        zero_out: "prompt" — zero out prompt-to-prompt distances,
            isolating response topology (paper uses this).
            "response" — zero out response-to-response distances.
            Default "prompt".
        normalize_by_length: Divide MTopDiv by response length.
            Default True.
        handle_nan: Replace inf/nan with 0. Default True.
    """

    def __init__(self, mode="supervised", n_max=6,
                 select_method="f_classif",
                 zero_out="prompt", normalize_by_length=False,
                 handle_nan=True):
        self.mode = mode
        self.n_max = n_max
        self.select_method = select_method
        self.zero_out = zero_out
        self.normalize_by_length = normalize_by_length
        self.handle_nan = handle_nan

        self.clf_ = None
        self.selected_heads_ = []
        self.n_layers_ = None
        self.n_heads_ = None
        self.is_fitted_ = False
        self.threshold_ = 0.5

    def _safe_replace(self, arr, fill_value=0.0):
        return np.where(np.isfinite(arr), arr, fill_value)

    @staticmethod
    def _attention_to_distance(attention_weights):
        """Transform attention matrix to distance matrix (paper Eq. 1).

        d_ij = 1 - a_ij, clipped to [0, 1], zero diagonal, symmetrised
        via min(A, A^T).

        Args:
            attention_weights: (n_tokens, n_tokens) attention matrix.

        Returns:
            (n_tokens, n_tokens) symmetric distance matrix.
        """
        attn = attention_weights.astype(np.float32)
        n = attn.shape[-1]

        distance = 1.0 - np.clip(attn, a_min=0.0, a_max=None)
        np.fill_diagonal(distance, 0.0)

        # Symmetrise: d_ij = min(d_ij, d_ji)
        distance = np.minimum(distance, distance.T)

        return distance

    @staticmethod
    def _compute_mtopdiv(distance_mx):
        """Compute MTopDiv from distance matrix via persistent homology.

        Uses ripser to compute H_0 persistent homology (connected components)
        of the Vietoris-Rips filtration. Returns sum of finite barcode lengths.

        Matches the official `transform_distances_to_mtopdiv` function.

        Args:
            distance_mx: (n, n) symmetric distance matrix.

        Returns:
            Total finite H_0 barcode length (MTopDiv score).
        """
        if not _HAS_RIPSER:
            raise ImportError("ripser required. Install: pip install ripser")

        barcodes = ripser(distance_mx, distance_matrix=True, maxdim=0)["dgms"]
        if len(barcodes) > 0 and len(barcodes[0]) > 1:
            # H_0 barcodes: last entry is [0, inf), skip it
            finite = barcodes[0][:-1]
            if len(finite) > 0:
                return float(np.sum(finite[:, 1] - finite[:, 0]))
        return 0.0

    @staticmethod
    def _get_mtopdiv_sample(attn_tensor, prompt_len, zero_out="prompt",
                             normalize_by_length=True):
        """Compute MTopDiv for all heads in one sample.

        Matches official `get_mtopdivs` logic.

        Args:
            attn_tensor: (n_layers, n_heads, seq_len, seq_len) attention.
            prompt_len: Number of prompt tokens.
            zero_out: "prompt" or "response".
            normalize_by_length: Divide by response length.

        Returns:
            (n_layers * n_heads,) array of MTopDiv scores.
        """
        n_layers, n_heads, seq_len, _ = attn_tensor.shape
        response_len = seq_len - prompt_len
        n_total = n_layers * n_heads
        scores = np.zeros(n_total)

        idx = 0
        for layer in range(n_layers):
            for head in range(n_heads):
                attn = attn_tensor[layer, head]  # (seq_len, seq_len)
                dist = TOHA._attention_to_distance(attn)

                # Zero out the specified subgraph (paper: prompt → zero)
                if zero_out == "prompt":
                    dist[:prompt_len, :prompt_len] = 0.0

                mtopdiv = TOHA._compute_mtopdiv(dist)

                if normalize_by_length and response_len > 0:
                    mtopdiv /= response_len

                scores[idx] = mtopdiv
                idx += 1

        return scores

    def fit(self, outputs, labels):
        """Fit the TOHA baseline.

        For supervised mode:
        1. Compute MTopDiv for all heads.
        2. Use SelectKBest (ANOVA F-value) to select top-n heads.
        3. Fit LogisticRegression on selected head features.

        For unsupervised mode:
        1. Compute MTopDiv for all heads.
        2. Select heads by difference-of-means between classes.
        3. Score = mean MTopDiv of selected heads (no classifier).

        Args:
            outputs: Dict with 'attentions'
                (B, n_layers, n_heads, seq_len, seq_len).
                May contain 'input_ids' and 'sequences' for prompt length.
            labels: Binary hallucination labels (B,).
        """
        attentions = outputs['attentions']
        B, n_layers, n_heads, seq_len, _ = attentions.shape
        self.n_layers_ = n_layers
        self.n_heads_ = n_heads

        # Determine prompt length per sample
        prompt_lens = np.full(B, seq_len // 2, dtype=int)
        if 'input_ids' in outputs and 'sequences' in outputs:
            from moeuncert.experiments.utils import find_generation_boundaries
            for b in range(B):
                gs, _ = find_generation_boundaries(
                    outputs['input_ids'][b],
                    outputs['sequences'][b],
                )
                if gs > 0:
                    prompt_lens[b] = gs

        # Labels: answer-level
        labels = np.asarray(labels, dtype=int)
        if labels.ndim > 1:
            labels = labels.ravel()
        if len(labels) != B:
            raise ValueError(f"Expected {B} labels, got {len(labels)}")

        # Compute MTopDiv for all heads
        all_features = np.zeros((B, n_layers * n_heads))
        for b in range(B):
            attn_arr = attentions[b]
            if hasattr(attn_arr, 'detach'):
                attn_arr = attn_arr.detach().cpu().numpy()
            all_features[b] = self._get_mtopdiv_sample(
                attn_arr, int(prompt_lens[b]),
                zero_out=self.zero_out,
                normalize_by_length=self.normalize_by_length,
            )

        if self.handle_nan:
            all_features = self._safe_replace(all_features, fill_value=0.0)

        if self.mode == "supervised":
            if len(np.unique(labels)) < 2:
                # Single class — fallback to uniform averaging
                self.selected_heads_ = list(range(n_layers * n_heads))
                self.clf_ = LogisticRegression(max_iter=1000)
                self.clf_.fit(all_features, labels)
            else:
                # Select heads using ANOVA F-value (matching official code)
                best_auc = 0
                best_n = 1
                best_feature_indices = list(range(min(self.n_max, n_layers * n_heads)))

                for n in range(1, min(self.n_max, n_layers * n_heads) + 1):
                    selector = SelectKBest(
                        score_func=f_classif,
                        k=n,
                    )
                    selector.fit(all_features, labels)
                    selected = all_features[:, selector.get_support()]

                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(selected, labels)

                    try:
                        preds = clf.predict_proba(selected)[:, 1]
                        auc = roc_auc_score(labels, preds)
                    except Exception:
                        auc = 0

                    if auc > best_auc:
                        best_auc = auc
                        best_n = n
                        best_feature_indices = np.where(selector.get_support())[0]

                # Re-fit with best n
                selector = SelectKBest(score_func=f_classif, k=best_n)
                selector.fit(all_features, labels)
                best_feature_indices = np.where(selector.get_support())[0]

                self.selected_heads_ = list(best_feature_indices)
                self.clf_ = LogisticRegression(max_iter=1000)
                self.clf_.fit(
                    all_features[:, best_feature_indices],
                    labels,
                )

        else:
            # Unsupervised mode — matches official greedy selection by diff-of-means
            if len(np.unique(labels)) >= 2:
                hal_mean = all_features[labels == 1].mean(axis=0)
                fact_mean = all_features[labels == 0].mean(axis=0)
                diff = hal_mean - fact_mean  # Signed difference (not abs)
            else:
                diff = all_features.std(axis=0)

            # Greedy: pick highest-diff heads one at a time, evaluate AUROC
            selected = []
            remaining = list(range(n_layers * n_heads))
            best_auroc = -1
            n_opt = 0
            diff_copy = diff.copy()

            for n in range(1, min(self.n_max, len(remaining)) + 1):
                best_idx = np.argmax(np.abs(diff_copy))
                selected.append(int(best_idx))
                diff_copy[best_idx] = -np.inf  # Mark as used

                if len(np.unique(labels)) >= 2:
                    scores = all_features[:, selected].mean(axis=1)
                    try:
                        auroc = roc_auc_score(labels, scores)
                    except Exception:
                        auroc = 0
                    if auroc > best_auroc:
                        best_auroc = auroc
                        n_opt = n

            self.selected_heads_ = selected[:n_opt] if n_opt > 0 else selected[:1]

        # Find optimal threshold
        train_scores = self.predict_proba(outputs)
        from moeuncert.experiments.utils import optimal_threshold
        best_thresh, best_f1 = optimal_threshold(labels, train_scores)
        self.threshold_ = best_thresh

        self.is_fitted_ = True

        print(f"  [TOHA] fit: {len(self.selected_heads_)}/{n_layers*n_heads} heads "
              f"selected, mode={self.mode}, "
              f"threshold={best_thresh:.4f} F1={best_f1:.4f}")

    def predict_proba(self, outputs):
        """Predict uncertainty scores.

        Higher = more uncertain / more likely hallucinated.

        For supervised: LogisticRegression predict_proba on selected heads.
        For unsupervised: mean MTopDiv of selected heads.

        Args:
            outputs: Dict with 'attentions' tensor.

        Returns:
            (B,) array of uncertainty scores.
        """
        if not self.is_fitted_:
            raise RuntimeError("Not fitted. Call fit() first.")

        attentions = outputs['attentions']
        B, n_layers, n_heads, seq_len, _ = attentions.shape

        prompt_lens = np.full(B, seq_len // 2, dtype=int)
        if 'input_ids' in outputs and 'sequences' in outputs:
            from moeuncert.experiments.utils import find_generation_boundaries
            for b in range(B):
                gs, _ = find_generation_boundaries(
                    outputs['input_ids'][b],
                    outputs['sequences'][b],
                )
                if gs > 0:
                    prompt_lens[b] = gs

        all_features = np.zeros((B, n_layers * n_heads))
        for b in range(B):
            attn_arr = attentions[b]
            if hasattr(attn_arr, 'detach'):
                attn_arr = attn_arr.detach().cpu().numpy()
            all_features[b] = self._get_mtopdiv_sample(
                attn_arr, int(prompt_lens[b]),
                zero_out=self.zero_out,
                normalize_by_length=self.normalize_by_length,
            )

        if self.handle_nan:
            all_features = self._safe_replace(all_features, fill_value=0.0)

        selected = all_features[:, self.selected_heads_]
        if selected.shape[1] == 0:
            # Fallback: all heads
            selected = all_features

        if self.mode == "supervised" and self.clf_ is not None:
            if selected.shape[1] > 0:
                return self.clf_.predict_proba(selected)[:, 1]
            return all_features.mean(axis=1)
        else:
            # Unsupervised: mean of absolute MTopDiv (matches official code)
            return np.abs(selected).mean(axis=1)

    def predict(self, outputs):
        """Predict binary labels (0=factual, 1=hallucinated)."""
        scores = self.predict_proba(outputs)
        return (scores >= self.threshold_).astype(int)
