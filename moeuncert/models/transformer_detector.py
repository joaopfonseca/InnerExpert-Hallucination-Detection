"""Transformer-encoder detector for per-token hallucination classification.

The model treats each (layer, feature_group) pair as a separate "token"
in a Transformer encoder, enabling self-attention across the model's
internal layers and feature types (attention scores, hidden scores,
router entropy, expert usage, etc.).

Architecture
------------
1. Input: (B, D) flat feature vector (e.g. 1344-dim for OLMoE).
2. Split into G groups by offset, reshape each to (B, n_layers, per_layer_size).
3. Per-group Linear projection (shared across layers within each group):
   (B, n_layers, per_layer_size) -> (B, n_layers, d_model).
4. Concatenate all groups: (B, total_tokens, d_model) where
   total_tokens = G * n_layers (e.g. 6 * 16 = 96 for OLMoE).
5. Add learnable positional encoding.
6. nn.TransformerEncoder (L layers, H heads, GELU, dropout).
7. Mean pool + LayerNorm -> (B, d_model).
8. Linear head -> (B, n_classes) logits.

Integration
------------
Used via ``skorch.NeuralNetClassifier`` so it is sklearn-compatible
(works with ``Pipeline``, ``GridSearchCV``, pickle save/load).

Example
-------
>>> from skorch import NeuralNetClassifier
>>> net = NeuralNetClassifier(
...     module=LayerGroupTransformerClassifier,
...     module__group_sizes=[(16, 1), (16, 16), (16, 1), (16, 1), (16, 1), (16, 64)],
...     module__d_model=64,
...     module__n_heads=4,
...     module__n_transformer_layers=2,
...     criterion=torch.nn.CrossEntropyLoss,
...     optimizer=torch.optim.Adam,
...     optimizer__lr=1e-3,
...     max_epochs=50,
... )
>>> net.fit(X_train, y_train)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature-layout computation
# ---------------------------------------------------------------------------

# Must match the ordering in 3.0-detection-model-training.py's merge_features().
_FEATURE_GROUP_ORDER = [
    "hidden_scores",
    "attention_scores",
    "router_entropy",
    "expert_hidden_scores",
    "expert_similarities",
    "expert_usage",
]


def compute_group_sizes(
    features: Dict[str, torch.Tensor],
    n_layers: int,
) -> List[Tuple[int, int]]:
    """Compute per-group ``(n_layers, per_layer_size)`` from the features dict.

    Called **before** ``merge_features`` flattens the structured tensors
    into a single 2D matrix.  The returned list is ordered to match
    ``merge_features``'s ``numerical_features`` list so that the flat
    vector's group offsets can be reconstructed.

    Parameters
    ----------
    features : Dict[str, torch.Tensor]
        Per-group feature tensors as produced by
        ``prepare_training_data``.  Each value has shape
        ``(n_tokens, total_for_group)`` where
        ``total_for_group = n_layers * per_layer_size``.
    n_layers : int
        Number of MoE layers in the model (e.g. 16 for OLMoE, 35 for
        Gemma 4).  Inferred from ``features["hidden_scores"].shape[1]``
        in the caller.

    Returns
    -------
    List[Tuple[int, int]]
        One ``(n_layers, per_layer_size)`` per feature group, in the
        same order as ``_FEATURE_GROUP_ORDER``.  Groups absent from
        ``features`` are skipped.
    """
    group_sizes: List[Tuple[int, int]] = []
    for key in _FEATURE_GROUP_ORDER:
        if key not in features:
            continue
        tensor = features[key]
        total = tensor.shape[1] if tensor.ndim >= 2 else 1
        if total % n_layers != 0:
            raise ValueError(
                f"Feature group '{key}' has {total} features, which is not "
                f"divisible by n_layers={n_layers}. Cannot infer per_layer_size."
            )
        per_layer_size = total // n_layers
        group_sizes.append((n_layers, per_layer_size))
    return group_sizes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LayerGroupTransformerClassifier(nn.Module):
    """Per-token hallucination classifier using a Transformer encoder
    over (layer, feature_group) tokens.

    Parameters
    ----------
    group_sizes : List[Tuple[int, int]]
        One ``(n_layers, per_layer_size)`` per feature group, ordered to
        match the flat input vector's group offsets.  Computed by
        ``compute_group_sizes()``.
    d_model : int
        Dimensionality of each token after projection (default 64).
    n_heads : int
        Number of attention heads in the Transformer encoder (default 4).
        Must evenly divide ``d_model``.
    n_transformer_layers : int
        Number of Transformer encoder layers (default 2).
    dropout : float
        Dropout rate in the Transformer encoder and classification head
        (default 0.1).
    n_classes : int
        Number of output classes (default 2 for binary classification).
    """

    def __init__(
        self,
        group_sizes: List[Tuple[int, int]],
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.group_sizes = group_sizes
        self.d_model = d_model

        # Per-group projections (shared across layers within each group).
        self.projections = nn.ModuleList([
            nn.Linear(per_layer_size, d_model)
            for (_, per_layer_size) in group_sizes
        ])

        # Total tokens = sum of n_layers across all groups.
        total_tokens = sum(n_layers for (n_layers, _) in group_sizes)
        self.total_tokens = total_tokens

        # Learnable positional encoding (one per token position).
        self.pos_encoding = nn.Parameter(
            torch.randn(1, total_tokens, d_model) * 0.02
        )

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            enable_nested_tensor=False,
        )

        # Classification head.
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Flat feature vector of shape ``(B, total_features)`` where
            ``total_features = sum(n_layers * per_layer_size)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, n_classes)``.
        """
        B = x.shape[0]
        tokens: List[torch.Tensor] = []
        offset = 0

        for i, (n_layers, per_layer_size) in enumerate(self.group_sizes):
            group_len = n_layers * per_layer_size
            group_flat = x[:, offset : offset + group_len]  # (B, n_layers * per_layer_size)
            group_reshaped = group_flat.reshape(B, n_layers, per_layer_size)
            projected = self.projections[i](group_reshaped)  # (B, n_layers, d_model)
            tokens.append(projected)
            offset += group_len

        # (B, total_tokens, d_model)
        token_seq = torch.cat(tokens, dim=1)
        token_seq = token_seq + self.pos_encoding

        # (B, total_tokens, d_model)
        encoded = self.transformer(token_seq)

        # Mean pool over all tokens -> (B, d_model)
        pooled = encoded.mean(dim=1)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        # (B, n_classes)
        logits = self.classifier(pooled)
        return logits