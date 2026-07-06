"""Model architectures for hallucination detection.

Currently provides the ``LayerGroupTransformerClassifier`` — a
Transformer encoder that treats each (layer, feature_group) pair as a
separate token, enabling self-attention across the model's internal
layers and feature types.
"""

from .transformer_detector import (
    LayerGroupTransformerClassifier,
    compute_group_sizes,
)

__all__ = [
    "LayerGroupTransformerClassifier",
    "compute_group_sizes",
]