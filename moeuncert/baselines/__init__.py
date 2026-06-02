"""
Baselines for hallucination detection and uncertainty estimation.

Each baseline follows a scikit-learn-style API with three core methods:
    fit(outputs, labels)      — calibrate thresholds or train the model
    predict(outputs)          — binary hallucination labels (0/1)
    predict_proba(outputs)    — continuous uncertainty scores (higher = more likely hallucinated)
"""

from ._base import BaseBaseline
from .predictive_entropy import PredictiveEntropy
from .llm_check import LLMCheck
from .selfcheck_gpt import SelfCheckNLI, SelfCheckPrompt
from .semantic_energy import SemanticEnergy, compute_semantic_energy
from .semantic_uncertainty import (
    SemanticUncertainty,
    EntailmentDeberta,
    get_semantic_ids,
    semantic_ids_to_groups,
    semantic_ids_to_clusters,
    logsumexp_by_id,
)
from .halunet import HaluNet

__all__ = [
    "BaseBaseline",
    "PredictiveEntropy",
    "LLMCheck",
    "SelfCheckNLI",
    "SelfCheckPrompt",
    "SemanticEnergy",
    "compute_semantic_energy",
    "SemanticUncertainty",
    "EntailmentDeberta",
    "get_semantic_ids",
    "semantic_ids_to_groups",
    "semantic_ids_to_clusters",
    "logsumexp_by_id",
    "HaluNet",
]