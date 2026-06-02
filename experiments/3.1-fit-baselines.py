"""
Fit baseline thresholds on training data.

This script loads multi-year training data (e.g., RealtimeQA 2022-2025),
extracts features for each threshold-based baseline, finds optimal thresholds
(and optimal layers for LLM-Check), and saves them to thresholds.json.

Baselines fitted:
- PredictiveEntropy: fits threshold on per-token entropy scores
- LLM-Check: fits threshold + selects optimal layer per score type
- SemanticUncertainty: fits threshold on semantic entropy (requires sampled data)
- SemanticEnergy: fits threshold on energy scores (requires sampled data)
- TokenMahalanobis: trains density model on hidden states + fits threshold
- TOHA: trains head-selection model on attention matrices + fits threshold

SelfCheckGPT requires no threshold fitting (pure inference).
HaluNet is trained separately (3.2-train-halunet.py).

Usage:
    python 3.1-fit-baselines.py --train-years 2022 2023 2024 2025
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from moeuncert.experiments import (
    resolve_model_slug,
    resolve_dataset_slug,
    load_multi_year_data,
    optimal_threshold,
    stratified_group_split,
)


def extract_answer_level_labels(df_labeled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract answer-level labels from labeled dataframe.
    
    Returns question_ids and binary labels (0=factual, 1=hallucinated).
    Uses LLM label when available, falling back to weak label per row.
    """
    if 'label_llm_answer' not in df_labeled.columns and 'label_weak_hallucination' not in df_labeled.columns:
        raise ValueError("No hallucination labels found in dataframe")

    llm_labels = df_labeled.get('label_llm_answer', pd.Series(dtype=float))
    weak_labels = df_labeled.get('label_weak_hallucination', pd.Series(dtype=float))

    # Per-row fallback: use LLM label if valid, otherwise weak label
    labels = llm_labels.where(llm_labels.notna(), weak_labels).astype(int).values
    qids = df_labeled['question_id'].values
    return qids, labels


def aggregate_token_to_answer(
    token_scores: np.ndarray,
    question_ids: np.ndarray,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-token scores to answer-level.
    
    Parameters
    ----------
    token_scores : np.ndarray
        Per-token scores.
    question_ids : np.ndarray
        Question ID for each token.
    aggregation : str
        'mean' or 'max'.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Unique question IDs and aggregated scores.
    """
    unique_qids = np.unique(question_ids)
    aggregated = []
    for qid in unique_qids:
        mask = question_ids == qid
        if aggregation == "mean":
            aggregated.append(np.nanmean(token_scores[mask]))
        elif aggregation == "max":
            aggregated.append(np.nanmax(token_scores[mask]))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return unique_qids, np.array(aggregated)


def fit_predictive_entropy(
    outputs: Dict[str, torch.Tensor],
    labels_df: pd.DataFrame,
    aggregation: str = "mean",
) -> Dict[str, float]:
    """Fit PredictiveEntropy baseline and return threshold."""
    print(f"\n--- PredictiveEntropy (aggregation={aggregation}) ---")
    
    # Extract pre-computed entropy from outputs. When compute_metrics was run
    # with k=None, `scores_entropy` is full-vocabulary entropy (not top-k).
    entropies = outputs['scores_entropy']  # (n_samples, seq_len)

    # Get question_ids and align with labels.
    # Use composite keys (question_id::evidence_present) to disambiguate base
    # vs RAG rows that share the same question_id.
    raw_qids = np.asarray(outputs['question_id'])
    ev_flags = np.asarray(outputs.get('evidence_present', [False] * len(raw_qids)))
    qids = np.array([f"{q}::{int(e)}" for q, e in zip(raw_qids, ev_flags)])

    # Aggregate to answer-level
    unique_qids, agg_scores = aggregate_token_to_answer(
        entropies.numpy().flatten(),
        np.repeat(qids, entropies.shape[1]),
        aggregation=aggregation,
    )

    # Match with labels using the same composite keys.
    label_qids, labels = extract_answer_level_labels(labels_df)
    ev_col = next(
        (c for c in ("evidence_present", "has_evidence", "with_evidence")
         if c in labels_df.columns),
        None,
    )
    if ev_col is not None:
        label_keys = np.array(
            [f"{q}::{int(e)}" for q, e in zip(label_qids, labels_df[ev_col].values)]
        )
    else:
        label_keys = label_qids
    qid_to_label = dict(zip(label_keys, labels))
    mask = np.array([qid in qid_to_label for qid in unique_qids])
    unique_qids = unique_qids[mask]
    agg_scores = agg_scores[mask]
    matched_labels = np.array([qid_to_label[qid] for qid in unique_qids])
    
    # Fit threshold using optimal_threshold utility (maximizes accuracy)
    threshold, f1 = optimal_threshold(matched_labels, agg_scores)
    
    # Compute AUROC for reference
    if len(np.unique(matched_labels)) > 1:
        auroc = roc_auc_score(matched_labels, agg_scores)
    else:
        auroc = 0.5
    
    print(f"  Optimal threshold: {threshold:.4f} (F1: {f1:.4f}, AUROC: {auroc:.4f})")
    
    return {
        "threshold": float(threshold),
        "aggregation": aggregation,
        "auroc": float(auroc),
    }


def fit_llm_check(
    outputs: Dict[str, torch.Tensor],
    labels_df: pd.DataFrame,
    score_type: str = "attention",
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """Fit LLM-Check baseline for a specific score type and find optimal layer."""
    print(f"\n--- LLM-Check (score_type={score_type}, aggregation={aggregation}) ---")

    # Build composite qids to disambiguate base vs RAG rows.
    raw_qids = np.asarray(outputs['question_id'])
    ev_flags = np.asarray(outputs.get('evidence_present', [False] * len(raw_qids)))
    qids = np.array([f"{q}::{int(e)}" for q, e in zip(raw_qids, ev_flags)])

    # Build label lookup with matching composite keys.
    label_qids, labels = extract_answer_level_labels(labels_df)
    ev_col = next(
        (c for c in ("evidence_present", "has_evidence", "with_evidence")
         if c in labels_df.columns),
        None,
    )
    if ev_col is not None:
        label_keys = np.array(
            [f"{q}::{int(e)}" for q, e in zip(label_qids, labels_df[ev_col].values)]
        )
    else:
        label_keys = label_qids
    qid_to_label = dict(zip(label_keys, labels))

    if score_type in ["attention", "hidden"]:
        # These have layer dimension - need to find optimal layer
        if score_type == "attention":
            # Shape: (n_samples, n_layers, n_heads, seq_len)
            scores = outputs['attention_scores']
            # Sum over heads to get (n_samples, n_layers, seq_len)
            scores = scores.sum(dim=2)
        elif score_type == "hidden":
            # Shape: (n_samples, seq_len, n_layers)
            scores = outputs['hidden_scores']
            scores = scores.permute(0, 2, 1)  # (n_samples, n_layers, seq_len)
        
        n_layers = scores.shape[1]
        
        # Evaluate each layer
        best_layer = 0
        best_auroc = 0.0
        
        for layer in range(n_layers):
            layer_scores = scores[:, layer, :].float().numpy()
            layer_scores = np.where(np.isfinite(layer_scores), layer_scores, np.nan).flatten()
            layer_qids = np.repeat(qids, scores.shape[2])

            unique_qids, agg_scores = aggregate_token_to_answer(
                layer_scores, layer_qids, aggregation=aggregation
            )
            mask = np.array([qid in qid_to_label for qid in unique_qids])
            layer_matched_labels = np.array([qid_to_label[qid] for qid in unique_qids[mask]])
            layer_agg_scores = agg_scores[mask]

            # Drop any NaN aggregates (all tokens were masked as -inf)
            nan_mask = ~np.isnan(layer_agg_scores)
            layer_matched_labels = layer_matched_labels[nan_mask]
            layer_agg_scores = layer_agg_scores[nan_mask]

            # Compute AUROC for this layer
            if len(layer_matched_labels) > 1 and len(np.unique(layer_matched_labels)) > 1:
                auroc = roc_auc_score(layer_matched_labels, layer_agg_scores)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
        
        print(f"  Best layer: {best_layer} (AUROC: {best_auroc:.4f})")

        # Re-compute scores for best layer and fit threshold.
        best_layer_scores = scores[:, best_layer, :].float().numpy()
        best_layer_scores = np.where(np.isfinite(best_layer_scores), best_layer_scores, np.nan).flatten()
        best_layer_qids = np.repeat(qids, scores.shape[2])
        unique_qids, agg_scores = aggregate_token_to_answer(
            best_layer_scores, best_layer_qids, aggregation=aggregation
        )
        mask = np.array([qid in qid_to_label for qid in unique_qids])
        unique_qids = unique_qids[mask]
        agg_scores = agg_scores[mask]
        matched_labels = np.array([qid_to_label[qid] for qid in unique_qids])

        # Drop any NaN aggregates before evaluating
        nan_mask = ~np.isnan(agg_scores)
        matched_labels = matched_labels[nan_mask]
        agg_scores = agg_scores[nan_mask]

        threshold, f1 = optimal_threshold(matched_labels, agg_scores)

        if len(matched_labels) > 1 and len(np.unique(matched_labels)) > 1:
            layer_auroc = roc_auc_score(matched_labels, agg_scores)
        else:
            layer_auroc = 0.5

        print(f"  Optimal threshold: {threshold:.4f} (F1: {f1:.4f}, AUROC: {layer_auroc:.4f})")
        
        return {
            "threshold": float(threshold),
            "layer": int(best_layer),
            "aggregation": aggregation,
            "auroc": float(best_auroc),
            "layer_auroc": float(layer_auroc),
        }
    
    elif score_type == "perplexity":
        # Answer-level directly — aggregation parameter is a no-op here
        # (perplexity is pre-computed by compute_metrics when return_baseline_features=True)
        perplexities = outputs['perplexity']  # (n_samples,)

        mask = np.array([qid in qid_to_label for qid in qids])
        matched_labels = np.array([qid_to_label[qid] for qid in qids[mask]])
        matched_perplexities = perplexities.numpy()[mask]

        threshold, f1 = optimal_threshold(matched_labels, matched_perplexities)
        auroc = roc_auc_score(matched_labels, matched_perplexities) if len(np.unique(matched_labels)) > 1 else 0.5
        
        return {
            "threshold": float(threshold),
            "aggregation": aggregation,
            "auroc": float(auroc),
        }
    
    elif score_type == "entropy":
        # Per-token entropy (pre-computed top-k entropy), aggregate to answer
        entropies = outputs['scores_entropy']  # (n_samples, seq_len)

        unique_qids, agg_scores = aggregate_token_to_answer(
            entropies.numpy().flatten(),
            np.repeat(qids, entropies.shape[1]),
            aggregation=aggregation,
        )
        mask = np.array([qid in qid_to_label for qid in unique_qids])
        unique_qids = unique_qids[mask]
        agg_scores = agg_scores[mask]
        matched_labels = np.array([qid_to_label[qid] for qid in unique_qids])

        threshold, f1 = optimal_threshold(matched_labels, agg_scores)
        auroc = roc_auc_score(matched_labels, agg_scores) if len(np.unique(matched_labels)) > 1 else 0.5

        return {
            "threshold": float(threshold),
            "aggregation": aggregation,
        }


def _semantic_ids_to_clusters(semantic_ids):
    """Convert flat cluster ID list to list-of-index-lists format."""
    cluster_map = {}
    for idx, cid in enumerate(semantic_ids):
        if cid not in cluster_map:
            cluster_map[cid] = []
        cluster_map[cid].append(idx)
    return list(cluster_map.values())


def fit_semantic_uncertainty(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    labels_df: pd.DataFrame,
    num_samples: int = 5,
    temperature: float = 0.7,
) -> Dict:
    """Fit SemanticUncertainty baseline using sampled responses.
    
    Loads sampled outputs, computes semantic entropy scores per question,
    and fits an F1-optimal threshold.
    """
    from moeuncert.experiments.data_loading import load_sampled_outputs
    from moeuncert.baselines.semantic_uncertainty import SemanticUncertainty

    print(f"\n--- SemanticUncertainty (num_samples={num_samples}) ---")

    # Load sampled data
    sampled = load_sampled_outputs(
        data_root, years, month, model, num_samples=num_samples
    )
    responses_by_qid = sampled["responses_by_qid"]
    log_probs_by_qid = sampled["log_probs_by_qid"]

    if not responses_by_qid:
        print("  WARNING: No sampled responses found.")
        return {"threshold": None, "note": "No sampled response data available"}

    # Extract labels
    label_qids, labels = extract_answer_level_labels(labels_df)
    qid_to_label = dict(zip(label_qids, labels))

    # Need to decode token IDs to text for NLI-based semantic clustering.
    # We'll load the tokenizer once.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Compute scores and align with labels
    su = SemanticUncertainty()
    scores = []
    matched_labels = []

    for qid, response_tokens_list in tqdm(responses_by_qid.items(), desc="  Computing SE"):
        if qid not in qid_to_label:
            continue

        # Decode token IDs to text strings for NLI
        responses = [
            tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in response_tokens_list
        ]
        log_probs = log_probs_by_qid.get(qid, [])

        if not responses or not log_probs:
            continue

        try:
            score = su.predict_proba(responses, log_probs)
            scores.append(score)
            matched_labels.append(qid_to_label[qid])
        except Exception as e:
            print(f"  WARNING: Failed to compute SE for {qid}: {e}")
            continue

    if not scores:
        print("  WARNING: Could not compute any Semantic Entropy scores.")
        return {"threshold": None, "note": "SE computation failed for all questions"}

    scores = np.array(scores)
    matched_labels = np.array(matched_labels)

    # Fit threshold using F1
    threshold, f1 = optimal_threshold(matched_labels, scores)
    auroc = roc_auc_score(matched_labels, scores) if len(np.unique(matched_labels)) > 1 else 0.5

    print(f"  Optimal threshold: {threshold:.4f} (F1: {f1:.4f}, AUROC: {auroc:.4f})")
    print(f"  Evaluated on {len(scores)} questions")

    return {
        "threshold": float(threshold),
        "auroc": float(auroc),
        "num_questions": int(len(scores)),
        "num_samples": num_samples,
    }


def fit_semantic_energy(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    labels_df: pd.DataFrame,
    num_samples: int = 5,
    temperature: float = 0.7,
) -> Dict:
    """Fit SemanticEnergy baseline using sampled responses.

    Loads sampled outputs, clusters responses via NLI, computes semantic
    energy scores per question, and fits an F1-optimal threshold.
    """
    from moeuncert.experiments.data_loading import load_sampled_outputs
    from moeuncert.baselines.semantic_uncertainty import SemanticUncertainty
    from moeuncert.baselines.semantic_energy import SemanticEnergy

    print(f"\n--- SemanticEnergy (num_samples={num_samples}) ---")

    # Load sampled data
    sampled = load_sampled_outputs(
        data_root, years, month, model, num_samples=num_samples
    )
    responses_by_qid = sampled["responses_by_qid"]
    log_probs_by_qid = sampled["log_probs_by_qid"]
    logits_by_qid = sampled["logits_by_qid"]

    if not responses_by_qid:
        print("  WARNING: No sampled responses found.")
        return {"threshold": None, "note": "No sampled response data available"}

    # Extract labels
    label_qids, labels = extract_answer_level_labels(labels_df)
    qid_to_label = dict(zip(label_qids, labels))

    # Load tokenizer for decoding and NLI clustering
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Use SemanticUncertainty's clustering for building semantic groups
    su = SemanticUncertainty()
    se = SemanticEnergy()

    scores = []
    matched_labels = []

    for qid, response_tokens_list in tqdm(responses_by_qid.items(), desc="  Computing SEnergy"):
        if qid not in qid_to_label:
            continue

        # Decode token IDs to text for NLI
        responses = [
            tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in response_tokens_list
        ]
        log_probs = log_probs_by_qid.get(qid, [])
        response_logits = logits_by_qid.get(qid, [])

        if not responses or not response_logits or len(responses) < 2:
            continue

        try:
            # Cluster responses using NLI
            semantic_ids = su.cluster_responses(responses)
            clusters = _semantic_ids_to_clusters(semantic_ids)

            # Compute probabilities per response (product of token probs)
            response_probs = [[np.exp(lp) for lp in ll] for ll in log_probs]

            score = se.predict_proba(
                response_logits=response_logits,
                response_probs=response_probs,
                clusters=clusters,
            )
            scores.append(score)
            matched_labels.append(qid_to_label[qid])
        except Exception as e:
            print(f"  WARNING: Failed to compute SEnergy for {qid}: {e}")
            continue

    if not scores:
        print("  WARNING: Could not compute any Semantic Energy scores.")
        return {"threshold": None, "note": "SE computation failed for all questions"}

    scores = np.array(scores)
    matched_labels = np.array(matched_labels)

    # Fit threshold using F1
    threshold, f1 = optimal_threshold(matched_labels, scores)
    auroc = roc_auc_score(matched_labels, scores) if len(np.unique(matched_labels)) > 1 else 0.5

    print(f"  Optimal threshold: {threshold:.4f} (F1: {f1:.4f}, AUROC: {auroc:.4f})")
    print(f"  Evaluated on {len(scores)} questions")

    return {
        "threshold": float(threshold),
        "auroc": float(auroc),
        "num_questions": int(len(scores)),
        "num_samples": num_samples,
    }


def fit_token_mahalanobis(
    outputs: Dict[str, torch.Tensor],
    labels_df: pd.DataFrame,
    aggregation: str = "mean",
    use_huq: bool = True,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Fit TokenMahalanobis baseline and return threshold.

    Faithful re-implementation of Vazhentsev et al. (NAACL 2025):
    computes per-layer MD from hidden states, trains Ridge on
    sequence-level MD features, optionally combines with MSP via HUQ.

    Args:
        outputs: Dict with 'hidden_states' tensor
            (B, seq_len, n_layers, hidden_size).
        labels_df: Labeled dataframe with hallucination labels.
        aggregation: 'mean' or 'max' for answer-level aggregation.
        use_huq: If True, use HUQ two-stage ranking (MD + MSP).
        alpha: Ridge regularization strength.

    Returns:
        Dict with threshold, aggregation, and AUROC.
    """
    print(f"\n--- TokenMahalanobis (aggregation={aggregation}, huq={use_huq}) ---")

    from moeuncert.baselines.token_mahalanobis import TokenMahalanobis

    raw_qids = np.asarray(outputs['question_id'])
    ev_flags = np.asarray(outputs.get('evidence_present', [False] * len(raw_qids)))
    qids = np.array([f"{q}::{int(e)}" for q, e in zip(raw_qids, ev_flags)])

    hidden_states = outputs['hidden_states']  # (B, seq_len, n_layers, hidden_size)
    B, seq_len, n_layers, hidden_size = hidden_states.shape

    # Build label lookup
    label_qids, labels = extract_answer_level_labels(labels_df)
    ev_col = next(
        (c for c in ("evidence_present", "has_evidence", "with_evidence")
         if c in labels_df.columns),
        None,
    )
    if ev_col is not None:
        label_keys = np.array(
            [f"{q}::{int(e)}" for q, e in zip(label_qids, labels_df[ev_col].values)]
        )
    else:
        label_keys = label_qids
    qid_to_label = dict(zip(label_keys, labels))

    # Build per-token labels (repeat answer label for each token)
    token_labels_list = []
    for i in range(B):
        qid = qids[i]
        label = qid_to_label.get(qid, -1)
        token_labels_list.append(np.full(seq_len, label, dtype=int))
    token_labels = np.concatenate(token_labels_list)

    valid_mask = token_labels >= 0
    if valid_mask.sum() == 0:
        print("  WARNING: No labeled tokens found for TokenMahalanobis.")
        return {"threshold": None, "note": "No labeled data"}

    fit_outputs = {'hidden_states': hidden_states}
    if 'scores' in outputs:
        fit_outputs['scores'] = outputs['scores']

    baseline = TokenMahalanobis(
        alpha=alpha,
        positive=True,
        use_huq=use_huq,
    )
    baseline.fit(fit_outputs, token_labels)

    # Predict answer-level scores
    scores = baseline.predict_proba(fit_outputs)  # (B,)

    # Match to labels
    matched_labels = np.array([qid_to_label[qid] for qid in qids])
    valid_answers = matched_labels >= 0

    if valid_answers.sum() == 0:
        return {"threshold": 0.5, "note": "No labeled answers"}

    threshold, f1 = optimal_threshold(matched_labels[valid_answers],
                                       scores[valid_answers])

    if len(np.unique(matched_labels[valid_answers])) > 1:
        auroc = roc_auc_score(matched_labels[valid_answers],
                              scores[valid_answers])
    else:
        auroc = 0.5

    print(f"  Optimal threshold: {threshold:.4f} (F1: {f1:.4f}, AUROC: {auroc:.4f})")

    return {
        "threshold": float(threshold),
        "aggregation": aggregation,
        "auroc": float(auroc),
        "use_huq": use_huq,
    }


def fit_toha(
    outputs: Dict[str, torch.Tensor],
    labels_df: pd.DataFrame,
    n_top_heads: int = 10,
) -> Dict[str, Any]:
    """Fit TOHA baseline and return threshold.

    Extracts attention matrices, computes topological features per head,
    selects heads that best discriminate hallucination, and finds the
    optimal threshold for answer-level predictions.

    Args:
        outputs: Dict with 'attentions' tensor
            (B, n_layers, n_heads, seq_len, seq_len).
        labels_df: Labeled dataframe with hallucination labels.
        n_top_heads: Number of top attention heads to select.

    Returns:
        Dict with threshold, n_selected_heads, and AUROC.
    """
    print(f"\n--- TOHA (n_top_heads={n_top_heads}) ---")

    from moeuncert.baselines.toha import TOHA

    raw_qids = np.asarray(outputs['question_id'])
    ev_flags = np.asarray(outputs.get('evidence_present', [False] * len(raw_qids)))
    qids = np.array([f"{q}::{int(e)}" for q, e in zip(raw_qids, ev_flags)])

    # Build label lookup
    label_qids, labels = extract_answer_level_labels(labels_df)
    ev_col = next(
        (c for c in ("evidence_present", "has_evidence", "with_evidence")
         if c in labels_df.columns),
        None,
    )
    if ev_col is not None:
        label_keys = np.array(
            [f"{q}::{int(e)}" for q, e in zip(label_qids, labels_df[ev_col].values)]
        )
    else:
        label_keys = label_qids
    qid_to_label = dict(zip(label_keys, labels))

    # Align labels with samples
    B = len(raw_qids)
    sample_labels = np.array([qid_to_label.get(qid, -1) for qid in qids])
    valid_mask = sample_labels >= 0

    if valid_mask.sum() == 0:
        print("  WARNING: No labeled samples found for TOHA.")
        return {"threshold": None, "note": "No labeled data"}

    # Prepare fit outputs (may also need input_ids/sequences for prompt_len)
    fit_outputs = {
        'attentions': outputs['attentions'],
    }
    if 'input_ids' in outputs and 'sequences' in outputs:
        fit_outputs['input_ids'] = outputs['input_ids']
        fit_outputs['sequences'] = outputs['sequences']

    baseline = TOHA(mode="supervised", n_max=n_top_heads)
    baseline.fit(fit_outputs, sample_labels[valid_mask])

    # Predict on all
    scores = baseline.predict_proba(fit_outputs)  # (B,)
    scores = np.asarray(scores)
    matched_labels = sample_labels[valid_mask]
    matched_scores = scores[valid_mask]

    threshold, f1 = optimal_threshold(matched_labels, matched_scores)

    if len(np.unique(matched_labels)) > 1:
        auroc = roc_auc_score(matched_labels, matched_scores)
    else:
        auroc = 0.5

    print(f"  Optimal threshold: {threshold:.4f} (F1: {f1:.4f}, AUROC: {auroc:.4f})")

    return {
        "threshold": float(threshold),
        "n_selected_heads": len(baseline.selected_heads_),
        "auroc": float(auroc),
        "n_top_heads": n_top_heads,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit baseline thresholds on training data"
    )
    
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
        help="Years to use for training (default: 2022 2023 2024 2025)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        choices=range(1, 13),
        metavar="MONTH",
        help="Month (1-12). Only valid when a single year is provided.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--label-model",
        type=str,
        default="zai-org/GLM-5.1",
        help="Label model name",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save thresholds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--aggregations",
        type=str,
        nargs="+",
        default=["mean", "max"],
        help="Aggregation methods for token-level baselines",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sampled responses per question for SU/SE baselines (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used during generation (default: 0.7)",
    )
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    model_slug = resolve_model_slug(args.model)
    save_dir = args.save_dir / model_slug
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'=' * 70}")
    print("FITTING BASELINE THRESHOLDS")
    print(f"{'=' * 70}")
    print(f"Train years: {args.train_years}")
    print(f"Model: {args.model}")
    print(f"Aggregations: {args.aggregations}")
    
    # Load training data
    df_labeled, outputs, _ = load_multi_year_data(
        args.data_root,
        args.train_years,
        args.month,
        args.model,
        args.label_model,
    )
    
    # Extract answer-level labels for reference
    qids, labels = extract_answer_level_labels(df_labeled)
    print(f"\nTotal answers: {len(labels)}")
    print(f"Hallucination rate: {labels.mean():.1%}")
    
    # Fit baselines
    thresholds = {}
    
    # PredictiveEntropy
    for agg in args.aggregations:
        thresholds[f"predictive_entropy_{agg}"] = fit_predictive_entropy(
            outputs, df_labeled, aggregation=agg
        )
    
    # LLM-Check (all score types)
    for score_type in ["attention", "hidden", "perplexity", "entropy"]:
        for agg in args.aggregations:
            if score_type in ["attention", "hidden", "entropy"]:
                key = f"llm_check_{score_type}_{agg}"
            else:
                key = f"llm_check_{score_type}"
            thresholds[key] = fit_llm_check(
                outputs, df_labeled, score_type=score_type, aggregation=agg
            )
    
    # SemanticUncertainty (requires sampled data from 1.1)
    thresholds["semantic_uncertainty"] = fit_semantic_uncertainty(
        args.data_root,
        args.train_years,
        args.month,
        args.model,
        df_labeled,
        num_samples=args.num_samples,
        temperature=getattr(args, 'temperature', 0.7),
    )
    
    # SemanticEnergy (requires sampled data from 1.1)
    thresholds["semantic_energy"] = fit_semantic_energy(
        args.data_root,
        args.train_years,
        args.month,
        args.model,
        df_labeled,
        num_samples=args.num_samples,
        temperature=getattr(args, 'temperature', 0.7),
    )
    
    # TokenMahalanobis
    for agg in args.aggregations:
        thresholds[f"token_mahalanobis_{agg}"] = fit_token_mahalanobis(
            outputs, df_labeled, aggregation=agg
        )

    # TOHA
    thresholds["toha"] = fit_toha(
        outputs, df_labeled
    )

    # SelfCheckGPT: no threshold needed
    thresholds["selfcheck_nli"] = {"note": "No threshold needed (pure inference)"}
    thresholds["selfcheck_prompt"] = {"note": "No threshold needed (pure inference)"}
    
    # Save thresholds
    thresholds_path = save_dir / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Thresholds saved to: {thresholds_path}")
    print(f"{'=' * 70}")
    
    # Print summary
    print("\nSummary:")
    for name, config in thresholds.items():
        if "threshold" in config and config["threshold"] is not None:
            print(f"  {name}: threshold={config['threshold']:.4f}", end="")
            if "layer" in config:
                print(f", layer={config['layer']}", end="")
            if "auroc" in config:
                print(f", AUROC={config['auroc']:.4f}", end="")
            print()
        else:
            print(f"  {name}: {config.get('note', 'N/A')}")


if __name__ == "__main__":
    main()
