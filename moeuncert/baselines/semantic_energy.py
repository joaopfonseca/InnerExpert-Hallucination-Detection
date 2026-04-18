"""
Semantic Energy baseline (Ma et al., 2025).

A sampling-based hallucination detection method that replaces the
probability-space entropy of Semantic Uncertainty with a logit-space
Boltzmann-inspired energy function. When all sampled responses cluster
together, Semantic Entropy gives 0 (confident), but the model's logits
may still indicate low confidence — Semantic Energy captures this signal.

The method:
1. Sample multiple responses per question (same as Semantic Uncertainty).
2. Cluster responses by semantic equivalence using an NLI model.
3. Compute per-cluster probability (proportion of responses in cluster)
   and per-cluster energy (negative mean logit of generated tokens).
4. Aggregate: Semantic Energy = Σ_cluster p(cluster) * E(cluster).

This baseline shares the sampling + clustering pipeline with Semantic
Uncertainty but uses logit magnitudes instead of probability entropy,
making it sensitive to model confidence even when responses are
semantically identical.

Paper: Semantic Energy: Detecting LLM Hallucination Beyond Entropy
       (arXiv 2508.14496)
Code: https://github.com/MaHuanAAA/SemanticEnergy
"""

import math
import numpy as np
import torch
from ._base import BaseBaseline


def _product_of_probs(probs_list):
    """Compute the product of token probabilities for each response.

    Args:
        probs_list: List of lists of token probabilities per response.
            Each inner list has shape (seq_len,).

    Returns:
        List of products (one float per response).
    """
    return [math.prod(sublist) for sublist in probs_list]


def _boltzmann_logits(logits_list):
    """Apply Boltzmann transformation: negative mean of logits per response.

    This follows the official implementation's `cal_boltzmann_logits`:
        logits_se = [-mean(sublist) for sublist in logits_list]

    Args:
        logits_list: List of lists of token logits per response.
            Each inner list has shape (seq_len,).

    Returns:
        List of energy values (one float per response).
    """
    return [-np.mean(sublist) for sublist in logits_list]


def _fermi_dirac(E, mu, kT=1.0):
    """Apply the Fermi-Dirac function.

    f(E) = E / (exp((E - mu) / kT) + 1)

    This is an alternative to the Boltzmann transformation that applies
    a smooth step function centered at mu. In the official code this is
    controlled by passing `fermi_mu` to `cal_flow`.

    Args:
        E: Energy level (logit value).
        mu: Chemical potential (center of the step).
        kT: Thermal energy (sharpness of the step, default 1.0).

    Returns:
        Transformed energy value.
    """
    return E / (math.exp((E - mu) / kT) + 1)


def _fermi_dirac_logits(logits_list, mu):
    """Apply Fermi-Dirac transformation to logits.

    For each response, compute the Fermi-Dirac function on each token
    logit, then take the mean.

    Args:
        logits_list: List of lists of token logits per response.
        mu: Chemical potential for the Fermi-Dirac function.

    Returns:
        List of transformed energy values (one float per response).
    """
    result = []
    for sublist in logits_list:
        transformed = [_fermi_dirac(l, mu) for l in sublist]
        result.append(np.mean(transformed))
    return result


def _sum_normalize(probs):
    """Normalize a list so elements sum to 1.

    Args:
        probs: List of probability values.

    Returns:
        Normalized list. Returns zeros if total is zero.
    """
    total = sum(probs)
    if total == 0:
        return [0.0 for _ in probs]
    return [p / total for p in probs]


def _cluster_cross_entropy(probs, logits, clusters):
    """Calculate cluster-level probabilities and energy scores.

    For each cluster:
    - Cluster probability: sum of normalized response probabilities
      (product of per-token probabilities) belonging to that cluster.
    - Cluster energy: negative sum of logits belonging to that cluster.

    This follows the official implementation's `cal_cluster_ce`.

    Args:
        probs: List of per-response probability products.
        logits: List of per-response energy values (negative mean logits
            for Boltzmann, or Fermi-Dirac transformed values).
        clusters: List of lists of response indices belonging to each
            cluster.

    Returns:
        Tuple of (probs_se, logits_se):
        - probs_se: List of cluster probabilities.
        - logits_se: List of cluster energy scores.
    """
    normalized_probs = _sum_normalize(probs)

    probs_se = []
    logits_se = []

    for cluster in clusters:
        cluster_prob_sum = sum(normalized_probs[i] for i in cluster)
        probs_se.append(cluster_prob_sum)

        # Official code sums (negated) logits within cluster:
        # cluster_logit_sum = -sum(logits[i] for i in cluster)
        # But logits here are already negative-mean (from _boltzmann_logits),
        # so we sum the energy values directly.
        cluster_logit_sum = sum(logits[i] for i in cluster)
        logits_se.append(cluster_logit_sum)

    return probs_se, logits_se


def compute_semantic_energy(
    response_logits,
    response_probs=None,
    clusters=None,
    semantic_groups=None,
    fermi_mu=None,
):
    """Compute Semantic Energy for a single question.

    This is the core computation matching the official implementation's
    `cal_flow` function.

    Two input modes are supported:

    Mode 1 (clusters): Provide pre-computed cluster assignments.
        clusters = [[0, 2], [1, 3]]  — responses 0,2 in cluster 0, etc.

    Mode 2 (semantic_groups): Provide boolean groups (same shape as
        Semantic Uncertainty). semantic_groups[i][j] = True means
        response i is semantically equivalent to response j.
        These will be converted to clusters internally.

    Args:
        response_logits: List of lists of token logits per response.
            Each inner list has shape (seq_len,). These are the raw
            logits (pre-softmax) of the generated tokens.
        response_probs: List of lists of token probabilities per response.
            Each inner list has shape (seq_len,). Required for computing
            cluster probabilities. If None, uniform weights are used.
        clusters: List of lists of response indices per cluster.
            Mutually exclusive with semantic_groups.
        semantic_groups: List of lists of booleans indicating semantic
            equivalence between responses. Mutually exclusive with clusters.
        fermi_mu: If provided, use Fermi-Dirac transformation instead of
            Boltzmann. The mu parameter controls the center of the step
            function.

    Returns:
        Tuple of (cluster_probs, cluster_energies):
        - cluster_probs: List of cluster probabilities.
        - cluster_energies: List of cluster energy scores.
    """
    if clusters is None and semantic_groups is None:
        raise ValueError("Must provide either clusters or semantic_groups.")

    if clusters is None:
        clusters = _semantic_groups_to_clusters(semantic_groups)

    # Compute per-response probability products
    if response_probs is not None:
        probs = _product_of_probs(response_probs)
    else:
        # Uniform weights if no probabilities provided
        probs = [1.0] * len(response_logits)

    # Compute per-response energy
    if fermi_mu is not None:
        logits = _fermi_dirac_logits(response_logits, fermi_mu)
    else:
        logits = _boltzmann_logits(response_logits)

    return _cluster_cross_entropy(probs, logits, clusters)


def _semantic_groups_to_clusters(semantic_groups):
    """Convert semantic equivalence groups to cluster assignments.

    semantic_groups is a list of lists where semantic_groups[i][j] = True
    means response i is semantically equivalent to response j.
    This is the format used by the Semantic Uncertainty implementation.

    Uses union-find to build disjoint clusters.

    Args:
        semantic_groups: List of lists of booleans, shape (N, N).

    Returns:
        List of lists of response indices per cluster.
    """
    n = len(semantic_groups)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if semantic_groups[i][j]:
                union(i, j)

    cluster_map = {}
    for i in range(n):
        root = find(i)
        if root not in cluster_map:
            cluster_map[root] = []
        cluster_map[root].append(i)

    return list(cluster_map.values())


class SemanticEnergy(BaseBaseline):
    """
    Semantic Energy baseline for hallucination detection.

    Replaces the probability-space entropy of Semantic Uncertainty with
    a logit-space Boltzmann-inspired energy function. When all sampled
    responses cluster together, Semantic Entropy gives 0 (confident), but
    the model's logits may still indicate low confidence — Semantic Energy
    captures this signal.

    Two modes:
    - Boltzmann (default): E = -mean(logits) per response, then aggregated
      across clusters.
    - Fermi-Dirac: E = mean(fermi_dirac(logits, mu)) per response. Use
      `fermi_mu` parameter to enable.

    This baseline requires:
    1. Multiple sampled responses per question.
    2. Semantic clustering of responses (via NLI or equivalence checking).
    3. Penultimate layer logits for each generated token.

    Paper: Semantic Energy: Detecting LLM Hallucination Beyond Entropy
           (arXiv 2508.14496)
    Code: https://github.com/MaHuanAAA/SemanticEnergy
    """

    def __init__(self, fermi_mu=None):
        """
        Args:
            fermi_mu: If provided, use Fermi-Dirac transformation instead
                of Boltzmann. The mu parameter controls the center of
                the step function. Default None (Boltzmann).
        """
        self.fermi_mu = fermi_mu
        self.threshold = None

    def predict_proba(self, response_logits, response_probs=None,
                      clusters=None, semantic_groups=None):
        """
        Compute Semantic Energy score for a single question.

        The final score is the energy of the largest cluster (by
        probability), matching the official implementation which uses
        the "best cluster" for evaluation.

        Args:
            response_logits: List of lists of token logits per response.
                Each inner list contains the raw logits (pre-softmax) of
                the generated tokens for that response.
            response_probs: List of lists of token probabilities per
                response. Required for computing cluster probabilities.
                If None, uniform weights are used.
            clusters: List of lists of response indices per cluster.
                Mutually exclusive with semantic_groups.
            semantic_groups: List of lists of booleans indicating semantic
                equivalence between responses. Mutually exclusive with
                clusters.

        Returns:
            float: Semantic Energy score. Higher = more likely hallucinated.
        """
        cluster_probs, cluster_energies = compute_semantic_energy(
            response_logits=response_logits,
            response_probs=response_probs,
            clusters=clusters,
            semantic_groups=semantic_groups,
            fermi_mu=self.fermi_mu,
        )

        # Official code: best_cluster = argmax of logits_se (energy scores)
        best_idx = int(np.argmax(cluster_energies))
        return cluster_energies[best_idx]

    def predict_proba_all_clusters(self, response_logits, response_probs=None,
                                   clusters=None, semantic_groups=None):
        """
        Compute Semantic Energy scores for all clusters.

        Returns both cluster probabilities and energies, useful for
        detailed analysis.

        Args:
            response_logits: List of lists of token logits per response.
            response_probs: List of lists of token probabilities per
                response. If None, uniform weights are used.
            clusters: List of lists of response indices per cluster.
            semantic_groups: List of lists of booleans indicating semantic
                equivalence between responses.

        Returns:
            Tuple of (cluster_probs, cluster_energies).
        """
        return compute_semantic_energy(
            response_logits=response_logits,
            response_probs=response_probs,
            clusters=clusters,
            semantic_groups=semantic_groups,
            fermi_mu=self.fermi_mu,
        )

    def fit(self, response_logits_list, response_probs_list,
            clusters_list, labels):
        """
        Find the optimal threshold for binary classification.

        Args:
            response_logits_list: List of per-question response logits.
                Each element is a list of lists (one inner list per response).
            response_probs_list: List of per-question response probabilities.
                Each element is a list of lists (one inner list per response).
                If None, uniform weights are used.
            clusters_list: List of per-question cluster assignments.
                Each element is a list of lists of response indices.
            labels: Binary hallucination labels (0=factual, 1=hallucinated).
                One label per question.
        """
        scores = []
        for i in range(len(response_logits_list)):
            probs = response_probs_list[i] if response_probs_list is not None else None
            score = self.predict_proba(
                response_logits=response_logits_list[i],
                response_probs=probs,
                clusters=clusters_list[i],
            )
            scores.append(score)

        scores = np.array(scores)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy().astype(float)
        else:
            labels_np = np.array(labels, dtype=float)

        best_threshold = 0.0
        best_accuracy = 0.0

        for threshold in np.linspace(scores.min(), scores.max(), 100):
            preds = (scores >= threshold).astype(int)
            accuracy = (preds == labels_np).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold

    def predict(self, response_logits, response_probs=None,
                clusters=None, semantic_groups=None):
        """
        Predict binary hallucination label for a single question.

        Args:
            response_logits: List of lists of token logits per response.
            response_probs: List of lists of token probabilities per
                response. If None, uniform weights are used.
            clusters: List of lists of response indices per cluster.
            semantic_groups: List of lists of booleans indicating semantic
                equivalence between responses.

        Returns:
            int: Binary label (0=factual, 1=hallucinated).
        """
        if self.threshold is None:
            raise RuntimeError("Must call fit() before predict().")

        score = self.predict_proba(
            response_logits=response_logits,
            response_probs=response_probs,
            clusters=clusters,
            semantic_groups=semantic_groups,
        )
        return int(score >= self.threshold)