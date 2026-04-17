"""
Baselines for hallucination detection and uncertainty estimation.

Each baseline follows a scikit-learn-style API with three core methods:
    fit(outputs, labels)      — calibrate thresholds or train the model
    predict(outputs)          — binary hallucination labels (0/1)
    predict_proba(outputs)    — continuous uncertainty scores (higher = more likely hallucinated)
"""


class BaseBaseline:
    """Abstract base class for hallucination detection baselines.

    All baselines must implement fit, predict, and predict_proba following
    the scikit-learn convention. Training-free methods use fit to find
    optimal decision thresholds on validation data.
    """

    def fit(self, outputs, labels):
        """
        Fit the baseline on validation data.

        For trainable methods, this trains the model.
        For training-free methods, this finds the optimal threshold
        for binary classification.

        Args:
            outputs: Model outputs (format depends on the baseline).
            labels: Ground truth hallucination labels.
        """
        raise NotImplementedError

    def predict(self, outputs):
        """
        Predict binary hallucination labels.

        Args:
            outputs: Model outputs (format depends on the baseline).

        Returns:
            Binary labels (0 = factual, 1 = hallucinated).
        """
        raise NotImplementedError

    def predict_proba(self, outputs):
        """
        Predict continuous uncertainty scores.

        Higher scores indicate higher likelihood of hallucination.

        Args:
            outputs: Model outputs (format depends on the baseline).

        Returns:
            Continuous uncertainty scores.
        """
        raise NotImplementedError