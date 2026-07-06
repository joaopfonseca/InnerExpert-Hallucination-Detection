"""Evaluation utilities: prediction functions and metrics.

Exposes the per-method prediction functions extracted from
``experiments/4.0-model-evaluation.py`` and the metric helpers
extracted from ``experiments/5.0-results-analysis.py`` so that both
the RealtimeQA pipeline (4.0/5.0) and the OOS cross-dataset pipeline
(7.0/8.0) share the same code path.
"""

from .predictions import (
    build_ground_truth,
    evaluate_detector,
    evaluate_halunet,
    evaluate_llm_check,
    evaluate_predictive_entropy,
    evaluate_selfcheck,
    evaluate_semantic_energy,
    evaluate_semantic_uncertainty,
)
from .metrics import compute_ece, compute_metrics_for_predictions

__all__ = [
    "build_ground_truth",
    "evaluate_detector",
    "evaluate_halunet",
    "evaluate_llm_check",
    "evaluate_predictive_entropy",
    "evaluate_selfcheck",
    "evaluate_semantic_energy",
    "evaluate_semantic_uncertainty",
    "compute_ece",
    "compute_metrics_for_predictions",
]