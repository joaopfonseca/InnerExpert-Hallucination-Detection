"""
SelfCheckGPT baseline (Manakul et al., EMNLP 2023).

A sampling-based hallucination detection method that compares multiple
stochastically generated responses to measure consistency. When an LLM
has knowledge of a concept, sampled responses are consistent; for
hallucinated facts, samples tend to diverge and contradict.

We implement two variants:
- NLI: Uses DeBERTa NLI model to detect contradictions. Best
  performance-computation tradeoff per the paper.
- Prompt: Uses LLM prompting to assess consistency. Strongest
  overall performance but computationally heavy.

Paper: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
       for Generative Large Language Models (EMNLP 2023)
Code: https://github.com/potsawee/selfcheckgpt
Package: pip install selfcheckgpt
"""

import numpy as np
import torch
from ._base import BaseBaseline


class SelfCheckNLI(BaseBaseline):
    """
    SelfCheckGPT with NLI variant.

    Uses DeBERTa-v3-large fine-tuned on MNLI to detect contradictions
    between the original response sentences and sampled passages. For each
    sentence, computes P(contradict|sentence, sample) averaged over samples.

    This variant provides the best performance-computation tradeoff per
    the paper (AUC-PR 92.50 on non-factual detection), and is more
    practical than the Prompt variant.
    """

    def __init__(self, device=None):
        from selfcheckgpt.modeling_selfcheck import SelfCheckNLI as _SelfCheckNLI

        # Monkey-patch: transformers 5.x removed DebertaV2Tokenizer.batch_encode_plus.
        # The old API accepted batch_text_or_text_pairs=[(sentence, passage)] for
        # NLI sentence-pair encoding. The modern __call__ uses text=/text_pair=
        # instead. This shim translates the old calling convention to the new one.
        # NOTE: must use text_pair (sentence-pair encoding with token_type_ids
        # distinguishing premise/hypothesis), NOT text_target (which encodes the
        # second sequence as seq2seq labels, producing a shape mismatch with
        # DeBERTa's classification head).
        from transformers import DebertaV2Tokenizer
        if not hasattr(DebertaV2Tokenizer, "batch_encode_plus"):
            def _batch_encode_plus_compat(self, batch_text_or_text_pairs=None, **kwargs):
                if batch_text_or_text_pairs is not None:
                    if (
                        isinstance(batch_text_or_text_pairs, list)
                        and len(batch_text_or_text_pairs) > 0
                        and isinstance(batch_text_or_text_pairs[0], tuple)
                    ):
                        text = [p[0] for p in batch_text_or_text_pairs]
                        text_pair = [p[1] for p in batch_text_or_text_pairs]
                        return self(text=text, text_pair=text_pair, **kwargs)
                return self(text=batch_text_or_text_pairs, **kwargs)
            DebertaV2Tokenizer.batch_encode_plus = _batch_encode_plus_compat

        if device is None:
            device = torch.device("cpu")
        self.device = device
        self._checker = _SelfCheckNLI(device=device)
        self.threshold = None

    def predict_proba(self, sentences, sampled_passages):
        """
        Compute per-sentence inconsistency scores using NLI contradiction.

        Args:
            sentences: List[str] — sentences from the response to evaluate.
            sampled_passages: List[str] — stochastically generated responses.

        Returns:
            numpy array of inconsistency scores with shape (num_sentences,).
            Higher = more likely hallucinated. Scores are in [0, 1].
        """
        return self._checker.predict(sentences, sampled_passages)

    def fit(self, sentences, sampled_passages, labels):
        """
        Find the optimal threshold for binary classification.

        Args:
            sentences: List[str] — sentences from the response.
            sampled_passages: List[str] — stochastically generated responses.
            labels: Binary hallucination labels (0=factual, 1=hallucinated).
        """
        scores = self.predict_proba(sentences, sampled_passages)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy().astype(float)
        else:
            labels_np = np.array(labels, dtype=float)

        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in np.linspace(scores.min(), scores.max(), 100):
            preds = (scores >= threshold).astype(int)
            accuracy = (preds == labels_np).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold

    def predict(self, sentences, sampled_passages):
        """
        Predict binary hallucination labels.

        Args:
            sentences: List[str] — sentences from the response.
            sampled_passages: List[str] — stochastically generated responses.

        Returns:
            numpy array of binary labels (0=factual, 1=hallucinated).
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict().")

        scores = self.predict_proba(sentences, sampled_passages)
        return (scores >= self.threshold).astype(int)


class SelfCheckPrompt(BaseBaseline):
    """
    SelfCheckGPT with LLM Prompt variant.

    Uses an LLM to assess whether each sentence is supported by each
    sampled passage via zero-shot prompting. This is the strongest
    SelfCheckGPT variant per the paper (AUC-PR 93.42 on non-factual
    detection), but is computationally expensive as it requires LLM
    inference for each (sentence, sample) pair.
    """

    def __init__(self, model=None, device=None):
        from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt as _SelfCheckLLMPrompt

        if device is None:
            device = torch.device("cpu")
        self.device = device
        kwargs = {"device": device}
        if model is not None:
            kwargs["model"] = model
        self._checker = _SelfCheckLLMPrompt(**kwargs)
        self.threshold = None

    def set_prompt_template(self, prompt_template):
        """Override the default prompt template."""
        self._checker.set_prompt_template(prompt_template)

    def predict_proba(self, sentences, sampled_passages, verbose=False):
        """
        Compute per-sentence inconsistency scores using LLM prompting.

        For each (sentence, sample) pair, the LLM is asked "Is the sentence
        supported by the context?" and maps Yes→0.0, No→1.0, N/A→0.5.
        The final score is the average across samples.

        Args:
            sentences: List[str] — sentences from the response to evaluate.
            sampled_passages: List[str] — stochastically generated responses.
            verbose: If True, show progress bar.

        Returns:
            numpy array of inconsistency scores with shape (num_sentences,).
            Higher = more likely hallucinated. Scores are in [0, 1].
        """
        return self._checker.predict(sentences, sampled_passages, verbose=verbose)

    def fit(self, sentences, sampled_passages, labels, verbose=False):
        """
        Find the optimal threshold for binary classification.

        Args:
            sentences: List[str] — sentences from the response.
            sampled_passages: List[str] — stochastically generated responses.
            labels: Binary hallucination labels (0=factual, 1=hallucinated).
            verbose: If True, show progress bar during scoring.
        """
        scores = self.predict_proba(sentences, sampled_passages, verbose=verbose)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy().astype(float)
        else:
            labels_np = np.array(labels, dtype=float)

        best_threshold = 0.5
        best_accuracy = 0.0

        for threshold in np.linspace(scores.min(), scores.max(), 100):
            preds = (scores >= threshold).astype(int)
            accuracy = (preds == labels_np).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold

    def predict(self, sentences, sampled_passages, verbose=False):
        """
        Predict binary hallucination labels.

        Args:
            sentences: List[str] — sentences from the response.
            sampled_passages: List[str] — stochastically generated responses.
            verbose: If True, show progress bar during scoring.

        Returns:
            numpy array of binary labels (0=factual, 1=hallucinated).
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict().")

        scores = self.predict_proba(sentences, sampled_passages, verbose=verbose)
        return (scores >= self.threshold).astype(int)