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

__all__ = [
    "BaseBaseline",
    "PredictiveEntropy",
    "LLMCheck",
    "SelfCheckNLI",
    "SelfCheckPrompt",
]