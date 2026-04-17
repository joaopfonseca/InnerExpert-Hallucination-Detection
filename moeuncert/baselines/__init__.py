"""
Baselines for hallucination detection and uncertainty estimation.

Each baseline exposes a consistent interface:
    compute_uncertainty(...) -> per-token uncertainty scores
"""

from .predictive_entropy import predictive_entropy

__all__ = [
    "predictive_entropy",
]