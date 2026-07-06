"""Metric helpers shared by 5.0 (RealtimeQA analysis) and 8.0 (OOS analysis).

Extracted verbatim from ``experiments/5.0-results-analysis.py`` so that
both the in-distribution and out-of-sample analysis scripts use the
same metric definitions.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width binning)."""
    if len(y_proba) == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (y_proba >= low) & (y_proba < high)
        if high == 1.0:
            in_bin = (y_proba >= low) & (y_proba <= high)
        prop = in_bin.mean()
        if prop > 0:
            avg_conf = y_proba[in_bin].mean()
            avg_acc = y_true[in_bin].mean()
            ece += prop * abs(avg_conf - avg_acc)
    return float(ece)


def compute_metrics_for_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUROC / AUPRC / F1 / Accuracy / TPR@1/5/10%FPR / ECE.

    Returns a dict with keys:
        auroc, auprc, f1, accuracy,
        tpr_at_1fpr, tpr_at_5fpr, tpr_at_10fpr,
        ece, threshold, n

    When ``y_true`` is empty or has a single class, degenerate
    defaults are returned (AUROC=0.5, everything else=0.0).
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "tpr_at_1fpr": 0.0,
            "tpr_at_5fpr": 0.0,
            "tpr_at_10fpr": 0.0,
            "ece": 0.0,
            "threshold": float(threshold),
            "n": int(len(y_true)),
        }

    y_pred = (y_proba >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    tpr_at_1 = float(np.interp(0.01, fpr, tpr))
    tpr_at_5 = float(np.interp(0.05, fpr, tpr))
    tpr_at_10 = float(np.interp(0.10, fpr, tpr))
    ece = compute_ece(y_true, y_proba)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "accuracy": float(acc),
        "tpr_at_1fpr": tpr_at_1,
        "tpr_at_5fpr": tpr_at_5,
        "tpr_at_10fpr": tpr_at_10,
        "ece": ece,
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }