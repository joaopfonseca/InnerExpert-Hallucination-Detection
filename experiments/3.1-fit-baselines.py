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
from typing import Dict, List, Optional, Tuple

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
    """
    # Use LLM label if available, otherwise weak label
    if 'label_llm_answer' in df_labeled.columns:
        labels = df_labeled['label_llm_answer'].astype(int).values
    elif 'label_weak_hallucination' in df_labeled.columns:
        labels = df_labeled['label_weak_hallucination'].astype(int).values
    else:
        raise ValueError("No hallucination labels found in dataframe")
    
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
            aggregated.append(token_scores[mask].mean())
        elif aggregation == "max":
            aggregated.append(token_scores[mask].max())
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
    
    # Extract scores from outputs
    scores_tensor = outputs['scores']  # (n_samples, seq_len, vocab_size)
    
    # Compute per-token entropy
    probs = torch.softmax(scores_tensor, dim=-1)
    entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (n_samples, seq_len)
    
    # Get question_ids and align with labels
    qids = outputs['question_id'].numpy()
    
    # Aggregate to answer-level
    unique_qids, agg_scores = aggregate_token_to_answer(
        entropies.numpy().flatten(),
        np.repeat(qids, entropies.shape[1]),
        aggregation=aggregation,
    )
    
    # Match with labels
    label_qids, labels = extract_answer_level_labels(labels_df)
    qid_to_label = dict(zip(label_qids, labels))
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
) -> Dict[str, any]:
    """Fit LLM-Check baseline for a specific score type and find optimal layer."""
    print(f"\n--- LLM-Check (score_type={score_type}, aggregation={aggregation}) ---")
    
    # Extract question_ids
    qids = outputs['question_id'].numpy()
    label_qids, labels = extract_answer_level_labels(labels_df)
    qid_to_label = dict(zip(label_qids, labels))
    
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
            layer_scores = scores[:, layer, :].numpy().flatten()
            layer_qids = np.repeat(qids, scores.shape[2])
            
            unique_qids, agg_scores = aggregate_token_to_answer(
                layer_scores, layer_qids, aggregation=aggregation
            )
            matched_labels = np.array([qid_to_label[qid] for qid in unique_qids])
            
            # Compute AUROC for this layer
            if len(np.unique(matched_labels)) > 1:
                auroc = roc_auc_score(matched_labels, agg_scores)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
        
        print(f"  Best layer: {best_layer} (AUROC: {best_auroc:.4f})")
        
        # Fit threshold on best layer using optimal_threshold
        threshold, f1 = optimal_threshold(matched_labels, agg_scores)
        
        if len(np.unique(matched_labels)) > 1:
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
        # (perplexity is inherently a single scalar per answer)
        scores = outputs['scores']  # (n_samples, seq_len, vocab_size)
        sequences = outputs['sequences']  # (n_samples, total_seq_len)
        
        # Get generated tokens
        input_len = outputs['input_ids'].shape[1]
        gen_sequences = sequences[:, input_len:]
        
        # Compute per-token log probs
        log_probs = torch.log_softmax(scores, dim=-1)
        
        # Gather log probs of generated tokens
        gen_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=gen_sequences.unsqueeze(-1)
        ).squeeze(-1)
        
        # Perplexity = exp(-mean(log_prob))
        perplexities = torch.exp(-gen_log_probs.mean(dim=1))
        
        matched_labels = np.array([qid_to_label[qid] for qid in qids])
        
        threshold, f1 = optimal_threshold(matched_labels, perplexities.numpy())
        auroc = roc_auc_score(matched_labels, perplexities.numpy()) if len(np.unique(matched_labels)) > 1 else 0.5
        
        return {
            "threshold": float(threshold),
            "aggregation": aggregation,
            "auroc": float(auroc),
        }
    
    elif score_type == "entropy":
        # Per-token entropy, aggregate to answer
        scores = outputs['scores']
        probs = torch.softmax(scores, dim=-1)
        entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        unique_qids, agg_scores = aggregate_token_to_answer(
            entropies.numpy().flatten(),
            np.repeat(qids, entropies.shape[1]),
            aggregation=aggregation,
        )
        matched_labels = np.array([qid_to_label[qid] for qid in unique_qids])
        
        threshold, f1 = optimal_threshold(matched_labels, agg_scores)
        auroc = roc_auc_score(matched_labels, agg_scores) if len(np.unique(matched_labels)) > 1 else 0.5
        
        return {
            "threshold": float(threshold),
            "aggregation": aggregation,
        }


def fit_semantic_uncertainty(
    sampled_outputs_dir: Path,
    labels_df: pd.DataFrame,
) -> Dict[str, float]:
    """Fit SemanticUncertainty baseline (requires sampled responses).
    
    TODO: Implement sampled data loading and SU fitting.
    """
    print("\n--- SemanticUncertainty (TODO: requires sampled responses) ---")
    return {"threshold": None, "note": "Requires sampled response data"}


def fit_semantic_energy(
    sampled_outputs_dir: Path,
    labels_df: pd.DataFrame,
) -> Dict[str, float]:
    """Fit SemanticEnergy baseline (requires sampled responses).
    
    TODO: Implement sampled data loading and SE fitting.
    """
    print("\n--- SemanticEnergy (TODO: requires sampled responses) ---")
    return {"threshold": None, "note": "Requires sampled response data"}


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
    df_labeled, outputs = load_multi_year_data(
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
    
    # SemanticUncertainty (requires sampled data)
    # TODO: Implement when sampled data is available
    thresholds["semantic_uncertainty"] = fit_semantic_uncertainty(
        args.data_root, df_labeled
    )
    
    # SemanticEnergy (requires sampled data)
    # TODO: Implement when sampled data is available
    thresholds["semantic_energy"] = fit_semantic_energy(
        args.data_root, df_labeled
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
