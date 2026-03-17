import numpy as np
from sklearn.metrics import roc_curve

def optimal_threshold(y_true, scores):
    """Return the score threshold that maximises accuracy."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    acc = (tpr * n_pos + (1 - fpr) * n_neg) / len(y_true)
    best = np.argmax(acc)
    return thresholds[best], acc[best]
