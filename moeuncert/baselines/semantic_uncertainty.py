"""
Semantic Uncertainty baseline (Kuhn et al., ICLR 2023 / Nature 2024).

The gold standard for training-free hallucination detection. Clusters multiple
sampled generations by semantic equivalence using an NLI model, then computes
entropy over the cluster distribution. When responses diverge into multiple
semantic clusters, entropy is high → likely hallucinating. When they cluster
together, entropy is low → likely factual.

The method addresses a key limitation of predictive entropy: "Paris" and
"the French capital" are semantically identical but token-different, so raw
entropy overestimates uncertainty. By clustering at the semantic level, SE
avoids this.

Paper: Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation
       in Natural Language Generation (ICLR 2023)
       Detecting Hallucinations in Large Language Models Using Semantic Entropy
       (Nature, 2024)
Code: https://github.com/jlko/semantic_uncertainty
"""

import numpy as np
import torch
from ._base import BaseBaseline


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic clusters using NLI.

    This follows the official implementation's algorithm:
    - For each pair of responses, check bidirectional entailment using NLI.
    - Two responses are "semantically equivalent" if:
      - Strict mode: both directions are entailment
      - Non-strict mode (default): no direction is contradiction AND not both
        are neutral

    Args:
        strings_list: List[str] — generated responses to cluster.
        model: An entailment checker with a check_implication(text1, text2)
            method returning 0 (contradiction), 1 (neutral), or 2 (entailment).
        strict_entailment: If True, require both directions to be entailment.
            If False (default), allow entailment in one direction.
        example: Optional dict with 'question' key, used by LLM-based
            entailment checkers.

    Returns:
        List[int] — cluster assignment IDs, one per response.
    """

    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)

        if strict_entailment:
            return (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            return (0 not in implications) and ([1, 1] != implications)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0

    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    return semantic_set_ids


def semantic_ids_to_groups(semantic_ids):
    """Convert semantic cluster IDs to a boolean equivalence matrix.

    Args:
        semantic_ids: List[int] — cluster assignment IDs from get_semantic_ids.

    Returns:
        List[List[bool]] — semantic_groups[i][j] = True iff responses i and j
        are in the same semantic cluster.
    """
    n = len(semantic_ids)
    groups = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if semantic_ids[i] == semantic_ids[j]:
                groups[i][j] = True
    return groups


def semantic_ids_to_clusters(semantic_ids):
    """Convert semantic cluster IDs to a list of index lists.

    Args:
        semantic_ids: List[int] — cluster assignment IDs from get_semantic_ids.

    Returns:
        List[List[int]] — clusters[k] = [indices of responses in cluster k].
    """
    cluster_map = {}
    for idx, cid in enumerate(semantic_ids):
        if cid not in cluster_map:
            cluster_map[cid] = []
        cluster_map[cid].append(idx)
    return list(cluster_map.values())


def logsumexp_by_id(semantic_ids, log_likelihoods):
    """Compute log probability of each semantic cluster using logsumexp.

    This follows the official implementation's `logsumexp_by_id` with
    agg='sum_normalized'.

    For each cluster c:
        log p(c) = logsumexp(log_likelihoods_in_c) - logsumexp(all_log_likelihoods)

    Args:
        semantic_ids: List[int] — cluster assignment IDs.
        log_likelihoods: List[float] — log p(response) for each response,
            where log p(response) = sum of log probabilities of each token.

    Returns:
        List[float] — log probability of each semantic cluster.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))

    log_likelihood_per_cluster = []
    total_logsumexp = np.log(np.sum(np.exp(log_likelihoods)))

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        log_lik_norm = id_log_likelihoods - total_logsumexp
        logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        log_likelihood_per_cluster.append(logsumexp_value)

    return log_likelihood_per_cluster


def compute_semantic_entropy(log_probs):
    """Compute Semantic Entropy from per-response log probabilities.

    This follows the official implementation:
    1. Compute per-response log-likelihood: log p(response) = sum of log probs
    2. Assign responses to semantic clusters via NLI
    3. Compute cluster probabilities via logsumexp normalization
    4. Compute entropy over cluster probabilities

    Args:
        log_probs: List[List[float]] — per-token log probabilities for each
            response. log_probs[i] is a list of log probabilities for the
            tokens in response i.
        semantic_ids: List[int] — cluster assignment IDs from get_semantic_ids.

    Returns:
        float — Semantic Entropy score. Higher = more uncertain.
    """
    # This function requires semantic_ids, so it's computed in the class method


class EntailmentDeberta:
    """NLI-based entailment checker using DeBERTa-v2-xlarge-mnli.

    Follows the official Semantic Uncertainty implementation. The model checks
    if text1 entails text2 (whether text2 follows from text1).

    Returns:
        0 = contradiction
        1 = neutral
        2 = entailment
    """

    def __init__(self, model_name="microsoft/deberta-v2-xlarge-mnli", device=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def check_implication(self, text1, text2, *args, **kwargs):
        """Check if text1 entails text2.

        Args:
            text1: Premise text.
            text2: Hypothesis text.

        Returns:
            int: 0 (contradiction), 1 (neutral), or 2 (entailment).
        """
        import torch.nn.functional as F

        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            largest_index = torch.argmax(F.softmax(logits, dim=1))

        return largest_index.cpu().item()


class SemanticUncertainty(BaseBaseline):
    """
    Semantic Uncertainty baseline for hallucination detection.

    Clusters multiple sampled responses by semantic equivalence using an NLI
    model, then computes entropy over the cluster distribution. When responses
    diverge into multiple semantic clusters, entropy is high → likely
    hallucinating. When they cluster together, entropy is low → likely factual.

    This is the gold standard for training-free hallucination detection
    (Kuhn et al., 2023, Nature 2024). Every reviewer will look for this.

    Requires:
    1. Multiple sampled responses per question (same as SelfCheckGPT and
       Semantic Energy).
    2. An NLI model for semantic clustering (default: DeBERTa-v2-xlarge-mnli).

    Paper: Semantic Uncertainty: Linguistic Invariances for Uncertainty
           Estimation in Natural Language Generation (ICLR 2023)
           Detecting Hallucinations in Large Language Models Using Semantic
           Entropy (Nature, 2024)
    Code: https://github.com/jlko/semantic_uncertainty
    """

    def __init__(self, entailment_model=None, strict_entailment=False):
        """
        Args:
            entailment_model: An object with a check_implication(text1, text2)
                method returning 0 (contradiction), 1 (neutral), or 2
                (entailment). If None, uses EntailmentDeberta by default.
            strict_entailment: If True, require both directions to be entailment
                for semantic equivalence. If False (default), responses are
                equivalent if neither direction is contradiction and not both
                are neutral.
        """
        if entailment_model is None:
            self.entailment_model = EntailmentDeberta()
        else:
            self.entailment_model = entailment_model
        self.strict_entailment = strict_entailment
        self.threshold = None

    def cluster_responses(self, responses, example=None):
        """Cluster responses by semantic equivalence.

        Args:
            responses: List[str] — generated responses for a single question.
            example: Optional dict with 'question' key for LLM-based checkers.

        Returns:
            List[int] — cluster assignment IDs, one per response.
        """
        return get_semantic_ids(
            responses,
            self.entailment_model,
            strict_entailment=self.strict_entailment,
            example=example,
        )

    def predict_proba(self, responses, log_probs, example=None):
        """
        Compute Semantic Entropy for a single question.

        Args:
            responses: List[str] — generated responses for a single question.
            log_probs: List[List[float]] — per-token log probabilities for each
                response. log_probs[i] is a list of log probabilities for
                the tokens in response i.
            example: Optional dict with 'question' key for LLM-based checkers.

        Returns:
            float — Semantic Entropy score. Higher = more uncertain.
        """
        semantic_ids = self.cluster_responses(responses, example=example)
        return self._compute_se_from_ids(log_probs, semantic_ids)

    def _compute_se_from_ids(self, log_probs, semantic_ids):
        """Compute Semantic Entropy from per-response log probs and cluster IDs.

        Args:
            log_probs: List[List[float]] — per-token log probabilities.
            semantic_ids: List[int] — cluster assignment IDs.

        Returns:
            float — Semantic Entropy score.
        """
        # Compute per-response log-likelihoods
        response_log_likelihoods = [np.sum(lp) for lp in log_probs]

        # Compute cluster probabilities via logsumexp normalization
        log_cluster_probs = logsumexp_by_id(semantic_ids, response_log_likelihoods)

        # Convert to probabilities
        cluster_probs = np.exp(log_cluster_probs)

        # Compute entropy over cluster probabilities
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-12))
        return entropy

    def fit(self, responses_list, log_probs_list, labels, example=None):
        """
        Find the optimal threshold for binary classification.

        Args:
            responses_list: List of per-question response lists.
            log_probs_list: List of per-question token log probabilities.
            labels: Binary hallucination labels (0=factual, 1=hallucinated).
            example: Optional dict with 'question' key.
        """
        scores = []
        for i in range(len(responses_list)):
            score = self.predict_proba(
                responses_list[i],
                log_probs_list[i],
                example=example,
            )
            scores.append(score)

        scores = np.array(scores)

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

    def predict(self, responses, log_probs, example=None):
        """
        Predict binary hallucination label for a single question.

        Args:
            responses: List[str] — generated responses.
            log_probs: List[List[float]] — per-token log probabilities.
            example: Optional dict with 'question' key.

        Returns:
            int — Binary label (0=factual, 1=hallucinated).
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict().")

        score = self.predict_proba(responses, log_probs, example=example)
        return int(score >= self.threshold)