"""
TOHA — TOpology-based HAllucination detector (Bazarova et al., 2025).

Uses topological divergence (persistent homology) on attention graphs to
detect hallucinations. Higher topological divergence between prompt and
response attention subgraphs indicates hallucination.

Paper: Hallucination Detection in LLMs with Topological Divergence on
       Attention Graphs (arXiv 2504.10063)
Code: https://github.com/sb-ai-lab/TOHA

Key idea: Transform attention weights to distance matrices, zero out
prompt-to-prompt distances, compute persistent homology (H_0 barcode)
via Vietoris-Rips filtration. The sum of barcode lengths (MTopDiv) is
the uncertainty score. Hallucinated responses show higher topological
divergence (less concentrated attention structure).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ._base import BaseBaseline

try:
    from ripser import ripser
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False


class TOHA(BaseBaseline):
    """TOHA baseline for hallucination detection.

    Computes MTopDiv (Manifold Topology Divergence) on attention graphs
    using persistent homology. For each attention head, transforms attention
    weights to distances, zeroes out prompt-to-prompt subgraph, computes
    H_0 barcode via Vietoris-Rips, and sums finite barcode lengths.

    In supervised mode, fits LogisticRegression on per-head MTopDiv scores.
    In unsupervised mode, averages MTopDiv across all heads.

    Args:
        mode: "supervised" (fit head selection + LR) or "unsupervised"
            (average all heads). Default "supervised".
        n_top_heads: Number of top heads to select in supervised mode.
            If None, uses all heads. Default None.
        zero_out: "prompt" or "response" — which subgraph to zero out.
            Per the paper, "prompt" isolates response topology.
            Default "prompt".
        normalize_by_length: Whether to divide MTopDiv by response length.
            Default True (matches paper).
        handle_nan: If True, replace inf/nan values with 0. Default True.
    """

    def __init__(self, mode="supervised", n_top_heads=None,
                 zero_out="prompt", normalize_by_length=True,
                 handle_nan=True):
        self.mode = mode
        self.n_top_heads = n_top_heads
        self.zero_out = zero_out
        self.normalize_by_length = normalize_by_length
        self.handle_nan = handle_nan

        self.clf_ = None
        self.head_weights_ = None
        self.selected_heads_ = None
        self.threshold_ = 0.5
        self.n_layers_ = None
        self.n_heads_ = None

    def _safe_replace(self, arr, fill_value=0.0):
        """Replace non-finite values with fill_value."""
        return np.where(np.isfinite(arr), arr, fill_value)

    @staticmethod
    def _attention_to_distance(attention_weights):
        """Transform attention matrix to distance matrix.

        Following the paper: d_ij = 1 - a_ij (clipped at 0),
        zero diagonal, symmetrised via min(A, A^T).

        Args:
            attention_weights: (n_tokens, n_tokens) attention matrix.

        Returns:
            (n_tokens, n_tokens) distance matrix.
        """
        attention_weights = attention_weights.astype(np.float32)
        n_tokens = attention_weights.shape[-1]

        # Distance = 1 - attention, clipped at 0
        distance_mx = 1.0 - np.clip(attention_weights, a_min=0.0, a_max=None)

        # Zero diagonal (no self-distance)
        zero_diag = np.ones((n_tokens, n_tokens)) - np.eye(n_tokens)
        distance_mx *= zero_diag

        # Symmetrise: d_ij = min(d_ij, d_ji)
        distance_mx = np.minimum(
            np.swapaxes(distance_mx, -1, -2),
            distance_mx,
        )
        return distance_mx

    @staticmethod
    def _compute_mtopdiv(distance_mx):
        """Compute MTopDiv from distance matrix using persistent homology.

        Computes H_0 barcode via Vietoris-Rips filtration and sums
        finite barcode lengths (birth - death).

        Args:
            distance_mx: (n_tokens, n_tokens) symmetric distance matrix.

        Returns:
            float: MTopDiv score (sum of finite H_0 barcode lengths).
        """
        if not _HAS_RIPSER:
            raise ImportError(
                "ripser is required for TOHA. Install with: pip install ripser"
            )

        barcodes = ripser(distance_mx, distance_matrix=True, maxdim=0)["dgms"]
        if len(barcodes) > 0 and len(barcodes[0]) > 1:
            # Sum of finite barcode lengths (birth - death for finite deaths)
            # barcodes[0] is H_0 diagram: [[birth, death], ...]
            # Last entry is typically [0, inf) — skip it
            finite_barcodes = barcodes[0][:-1]
            if len(finite_barcodes) > 0:
                lengths = finite_barcodes[:, 1] - finite_barcodes[:, 0]
                return float(np.sum(lengths))
        return 0.0

    @staticmethod
    def _get_mtopdivs_for_sample(attns, prompt_len, response_len,
                                  zero_out="prompt",
                                  normalize_by_length=True):
        """Compute MTopDiv for all heads in a single sample.

        Args:
            attns: (n_layers, n_heads, seq_len, seq_len) attention tensor.
            prompt_len: Length of prompt portion.
            response_len: Length of response portion.
            zero_out: "prompt" or "response".
            normalize_by_length: Divide by response length.

        Returns:
            (n_layers * n_heads,) array of MTopDiv scores.
        """
        n_layers, n_heads, seq_len, _ = attns.shape
        n_total = n_layers * n_heads
        mtopdivs = np.zeros(n_total)

        idx = 0
        for layer in range(n_layers):
            for head in range(n_heads):
                attn_head = attns[layer, head]  # (seq_len, seq_len)
                distance_mx = TOHA._attention_to_distance(attn_head)

                # Zero out the specified subgraph
                if zero_out == "prompt":
                    # Zero prompt-to-prompt distances
                    prompt_end = prompt_len
                    distance_mx[:prompt_end, :prompt_end] = 0.0
                elif zero_out == "response":
                    # Zero response-to-response distances
                    resp_start = prompt_len
                    distance_mx[resp_start:, resp_start:] = 0.0

                mtopdiv = TOHA._compute_mtopdiv(distance_mx)
                if normalize_by_length and response_len > 0:
                    mtopdiv /= response_len
                mtopdivs[idx] = mtopdiv
                idx += 1

        return mtopdivs

    def fit(self, outputs, labels):
        """Fit TOHA baseline.

        In supervised mode: compute MTopDiv for all heads, select top
        discriminative heads via AUROC, optionally fit LogisticRegression.
        In unsupervised mode: simply store head weights uniformly.

        Args:
            outputs: Dict with 'attentions' tensor of shape
                (batch_size, n_layers, n_heads, seq_len, seq_len).
                May also contain 'input_ids' and 'sequences' for prompt
                length detection.
            labels: Binary hallucination labels. Can be per-answer (B,)
                or per-token (B, seq_len). Answer-level used for scoring.
        """
        attentions = outputs['attentions']
        B, n_layers, n_heads, seq_len, _ = attentions.shape
        self.n_layers_ = n_layers
        self.n_heads_ = n_heads

        # Determine prompt length per sample
        prompt_lens = np.zeros(B, dtype=int)
        if 'input_ids' in outputs and 'sequences' in outputs:
            from moeuncert.experiments.utils import find_generation_boundaries
            for b in range(B):
                gs, _ = find_generation_boundaries(
                    outputs['input_ids'][b],
                    outputs['sequences'][b],
                )
                prompt_lens[b] = gs if gs > 0 else seq_len // 2
        else:
            prompt_lens = np.full(B, seq_len // 2, dtype=int)

        response_lens = seq_len - prompt_lens

        # Compute MTopDiv for all heads, all samples
        all_features = np.zeros((B, n_layers * n_heads))
        for b in range(B):
            attn_array = attentions[b]
            if hasattr(attn_array, 'detach'):
                attn_array = attn_array.detach().cpu().numpy()
            plen = int(prompt_lens[b])
            rlen = int(response_lens[b])
            all_features[b] = self._get_mtopdivs_for_sample(
                attn_array, plen, rlen,
                zero_out=self.zero_out,
                normalize_by_length=self.normalize_by_length,
            )

        if self.handle_nan:
            all_features = self._safe_replace(all_features, fill_value=0.0)

        # Handle labels (answer-level)
        labels = np.asarray(labels, dtype=int)
        if labels.ndim == 2:
            # Per-token: aggregate to answer-level
            labels = labels.max(axis=1)
        elif labels.ndim == 1 and len(labels) != B:
            if len(labels) == B * seq_len:
                labels = labels.reshape(B, seq_len).max(axis=1)
            else:
                raise ValueError(
                    f"Labels length {len(labels)} doesn't match batch size {B}"
                )

        # Head selection via AUROC (supervised)
        if self.mode == "supervised":
            head_auroc = np.zeros(n_layers * n_heads)
            for h in range(n_layers * n_heads):
                scores = all_features[:, h]
                if len(np.unique(labels)) > 1 and len(np.unique(scores)) > 1:
                    try:
                        head_auroc[h] = abs(roc_auc_score(labels, scores) - 0.5) * 2
                    except Exception:
                        head_auroc[h] = 0.0

            # Select top heads
            if self.n_top_heads is not None and self.n_top_heads < len(head_auroc):
                top_indices = np.argsort(head_auroc)[-self.n_top_heads:]
                weights = np.zeros_like(head_auroc)
                weights[top_indices] = head_auroc[top_indices]
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                self.head_weights_ = weights
                self.selected_heads_ = [
                    (int(idx) // n_heads, int(idx) % n_heads)
                    for idx in top_indices
                ]
            else:
                weights = head_auroc.copy()
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                self.head_weights_ = weights
                self.selected_heads_ = [
                    (l, h) for l in range(n_layers)
                    for h in range(n_heads)
                ]

            # Fit LogisticRegression on selected head features
            if len(self.selected_heads_) > 0:
                selected_features = all_features[:, self.head_weights_ > 0]
                if selected_features.shape[1] > 0:
                    self.clf_ = LogisticRegression(max_iter=1000)
                    self.clf_.fit(selected_features, labels)
        else:
            # Unsupervised: uniform weights
            self.head_weights_ = np.ones(n_layers * n_heads) / (n_layers * n_heads)
            self.selected_heads_ = [
                (l, h) for l in range(n_layers)
                for h in range(n_heads)
            ]

        # Find optimal threshold
        if self.mode == "supervised" and self.clf_ is not None:
            selected_features = all_features[:, self.head_weights_ > 0]
            if selected_features.shape[1] > 0:
                train_scores = self.clf_.predict_proba(selected_features)[:, 1]
            else:
                train_scores = all_features @ self.head_weights_
        else:
            train_scores = all_features @ self.head_weights_

        from moeuncert.experiments.utils import optimal_threshold
        best_thresh, best_f1 = optimal_threshold(labels, train_scores)
        self.threshold_ = best_thresh

        print(
            f"  [TOHA] fit complete: {n_layers} layers × {n_heads} heads, "
            f"{len(self.selected_heads_)} selected, mode={self.mode}"
        )
        print(f"  [TOHA] optimal threshold: {best_thresh:.4f} (F1: {best_f1:.4f})")

    def predict_proba(self, outputs):
        """Predict uncertainty scores.

        Args:
            outputs: Dict with 'attentions' tensor of shape
                (batch_size, n_layers, n_heads, seq_len, seq_len).

        Returns:
            (batch_size,) array of uncertainty scores (higher = more
            likely hallucinated).
        """
        if self.head_weights_ is None:
            raise RuntimeError("TOHA not fitted yet. Call fit() first.")

        attentions = outputs['attentions']
        B, n_layers, n_heads, seq_len, _ = attentions.shape

        # Determine prompt length
        prompt_lens = np.zeros(B, dtype=int)
        if 'input_ids' in outputs and 'sequences' in outputs:
            from moeuncert.experiments.utils import find_generation_boundaries
            for b in range(B):
                gs, _ = find_generation_boundaries(
                    outputs['input_ids'][b],
                    outputs['sequences'][b],
                )
                prompt_lens[b] = gs if gs > 0 else seq_len // 2
        else:
            prompt_lens = np.full(B, seq_len // 2, dtype=int)

        response_lens = seq_len - prompt_lens

        # Compute MTopDiv for all heads
        all_features = np.zeros((B, n_layers * n_heads))
        for b in range(B):
            attn_array = attentions[b]
            if hasattr(attn_array, 'detach'):
                attn_array = attn_array.detach().cpu().numpy()
            plen = int(prompt_lens[b])
            rlen = int(response_lens[b])
            all_features[b] = self._get_mtopdivs_for_sample(
                attn_array, plen, rlen,
                zero_out=self.zero_out,
                normalize_by_length=self.normalize_by_length,
            )

        if self.handle_nan:
            all_features = self._safe_replace(all_features, fill_value=0.0)

        # Predict using selected heads
        if self.mode == "supervised" and self.clf_ is not None:
            selected_features = all_features[:, self.head_weights_ > 0]
            if selected_features.shape[1] == 0:
                # Fallback to weighted average
                scores = all_features @ self.head_weights_
            else:
                scores = self.clf_.predict_proba(selected_features)[:, 1]
        else:
            scores = all_features @ self.head_weights_

        return scores

    def predict(self, outputs):
        """Predict binary hallucination labels.

        Args:
            outputs: Dict with 'attentions' tensor.

        Returns:
            (batch_size,) array of binary labels.
        """
        scores = self.predict_proba(outputs)
        return (scores >= self.threshold_).astype(int)
