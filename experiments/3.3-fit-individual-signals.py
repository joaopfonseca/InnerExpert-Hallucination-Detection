"""Fit thresholds for individual MoE signals on training data.

For each of the 6 individual MoE signals, this script:
  1. Loads train data via stream_multi_year_data (keeping only the relevant signal key)
  2. For each aggregation (mean, max):
     a. Loops over layers, computes per-layer AUROC on train data
     b. Selects the best layer
     c. Fits the optimal F1 threshold on that layer's scores
  3. Appends the results to models/<model_slug>/thresholds.json

Signals fitted:
  - router_entropy: entropy of router weight distribution per token/layer
  - expert_hidden_scores: SVD-based score from expert hidden states
  - expert_similarities: weighted cosine similarity among expert hidden states
  - expert_usage_entropy: entropy of expert usage distribution
  - expert_usage_gini: Gini impurity of expert usage distribution
  - expert_usage_effective_experts: inverse Herfindahl index of expert usage

Usage:
    python 3.3-fit-individual-signals.py --train-years 2024 2025
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from moeuncert.experiments import (
    resolve_model_slug,
    load_tokenizer_for_data,
    optimal_threshold,
    safe_roc_auc_score,
    stream_multi_year_data,
)
from moeuncert.experiments.data_loading import _concat_parts


SIGNALS = [
    "router_entropy",
    "expert_hidden_scores",
    "expert_similarities",
    "expert_usage_entropy",
    "expert_usage_gini",
    "expert_usage_effective_experts",
]


def extract_answer_level_labels(df_labeled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if 'label_llm_answer' not in df_labeled.columns and 'label_weak_hallucination' not in df_labeled.columns:
        raise ValueError("No hallucination labels found in dataframe")

    llm_labels = df_labeled.get('label_llm_answer', pd.Series(dtype=float))
    weak_labels = df_labeled.get('label_weak_hallucination', pd.Series(dtype=float))
    labels = llm_labels.where(llm_labels.notna(), weak_labels).astype(int).values
    qids = df_labeled['question_id'].values
    return qids, labels


def aggregate_token_to_answer(
    token_scores: np.ndarray,
    question_ids: np.ndarray,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
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


def fit_individual_signal(
    outputs: Dict[str, torch.Tensor],
    df_labeled: pd.DataFrame,
    signal_name: str,
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """Fit threshold + select best layer for a single individual MoE signal.

    Mirrors fit_llm_check's layer-selection logic from 3.1.
    """
    print(f"\n--- Signal: {signal_name} (aggregation={aggregation}) ---")

    if signal_name not in outputs:
        print(f"  WARNING: '{signal_name}' not in outputs. Skipping.")
        return {"threshold": None, "note": f"{signal_name} not in outputs"}

    raw_qids = np.asarray(outputs['question_id'])
    ev_flags = np.asarray(outputs.get('evidence_present', [False] * len(raw_qids)))
    qids = np.array([f"{q}::{int(e)}" for q, e in zip(raw_qids, ev_flags)])

    label_qids, labels = extract_answer_level_labels(df_labeled)
    ev_col = next(
        (c for c in ("evidence_present", "has_evidence", "with_evidence")
         if c in df_labeled.columns),
        None,
    )
    if ev_col is not None:
        label_keys = np.array(
            [f"{q}::{int(e)}" for q, e in zip(label_qids, df_labeled[ev_col].values)]
        )
    else:
        label_keys = label_qids
    qid_to_label = dict(zip(label_keys, labels))

    signal = outputs[signal_name]
    if signal.ndim != 3:
        print(f"  WARNING: {signal_name} has shape {signal.shape}, expected 3D (B, S, L). Skipping.")
        return {"threshold": None, "note": f"unexpected shape {signal.shape}"}

    n_layers = signal.shape[2]

    best_layer = 0
    best_auroc = 0.0

    for layer in range(n_layers):
        layer_scores = signal[:, :, layer].float().numpy()
        layer_scores = np.where(np.isfinite(layer_scores), layer_scores, np.nan).flatten()
        layer_qids = np.repeat(qids, signal.shape[1])

        unique_qids, agg_scores = aggregate_token_to_answer(
            layer_scores, layer_qids, aggregation=aggregation
        )
        mask = np.array([qid in qid_to_label for qid in unique_qids])
        if mask.sum() == 0:
            continue
        layer_matched_labels = np.array([qid_to_label[qid] for qid in unique_qids[mask]])
        layer_agg_scores = agg_scores[mask]

        nan_mask = ~np.isnan(layer_agg_scores)
        layer_matched_labels = layer_matched_labels[nan_mask]
        layer_agg_scores = layer_agg_scores[nan_mask]

        if len(layer_matched_labels) > 1 and len(np.unique(layer_matched_labels)) > 1:
            auroc = safe_roc_auc_score(layer_matched_labels, layer_agg_scores)
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer

    print(f"  Best layer: {best_layer} (AUROC: {best_auroc:.4f})")

    best_layer_scores = signal[:, :, best_layer].float().numpy()
    best_layer_scores = np.where(np.isfinite(best_layer_scores), best_layer_scores, np.nan).flatten()
    best_layer_qids = np.repeat(qids, signal.shape[1])
    unique_qids, agg_scores = aggregate_token_to_answer(
        best_layer_scores, best_layer_qids, aggregation=aggregation
    )
    mask = np.array([qid in qid_to_label for qid in unique_qids])
    unique_qids = unique_qids[mask]
    agg_scores = agg_scores[mask]
    matched_labels = np.array([qid_to_label[qid] for qid in unique_qids])

    nan_mask = ~np.isnan(agg_scores)
    matched_labels = matched_labels[nan_mask]
    agg_scores = agg_scores[nan_mask]

    threshold, f1 = optimal_threshold(matched_labels, agg_scores)

    if len(matched_labels) > 1 and len(np.unique(matched_labels)) > 1:
        layer_auroc = safe_roc_auc_score(matched_labels, agg_scores)
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


def main():
    parser = argparse.ArgumentParser(
        description="Fit thresholds for individual MoE signals on training data"
    )

    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=[2024, 2025],
        help="Years to use for training (default: 2024 2025)",
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
        help="Directory to save thresholds (models/<model_slug>/thresholds.json)",
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
        help="Aggregation methods for token-level signals",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    model_slug = resolve_model_slug(args.model)
    save_dir = args.save_dir / model_slug
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("FITTING INDIVIDUAL MoE SIGNAL THRESHOLDS")
    print(f"{'=' * 70}")
    print(f"Train years: {args.train_years}")
    print(f"Model: {args.model}")
    print(f"Signals: {SIGNALS}")
    print(f"Aggregations: {args.aggregations}")

    print("\n[streaming] Loading multi-year data one year at a time...")
    _load_tokenizer = load_tokenizer_for_data(args.model)

    filter_keys = {"question_id", "evidence_present"} | set(SIGNALS)

    per_year_dfs: List[pd.DataFrame] = []
    per_year_outputs: List[Dict] = []
    for df, outputs, _data_dir in stream_multi_year_data(
        args.data_root,
        args.train_years,
        args.month,
        args.model,
        args.label_model,
        pad_token_id=_load_tokenizer.pad_token_id,
        filter_keys=filter_keys,
    ):
        per_year_dfs.append(df)
        per_year_outputs.append(outputs)

    df_labeled = pd.concat(per_year_dfs, ignore_index=True)
    if len(per_year_outputs) == 1:
        outputs = per_year_outputs[0]
    else:
        outputs = _concat_parts(per_year_outputs, pad_token_id=_load_tokenizer.pad_token_id)
    del per_year_dfs, per_year_outputs

    qids, labels = extract_answer_level_labels(df_labeled)
    print(f"\nTotal answers: {len(labels)}")
    print(f"Hallucination rate: {labels.mean():.1%}")

    thresholds_path = save_dir / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
        print(f"\nLoaded existing thresholds from {thresholds_path}")
    else:
        thresholds = {}

    for signal_name in SIGNALS:
        for agg in args.aggregations:
            key = f"individual_signal_{signal_name}_{agg}"
            thresholds[key] = fit_individual_signal(
                outputs, df_labeled, signal_name=signal_name, aggregation=agg
            )

    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Thresholds saved to: {thresholds_path}")
    print(f"{'=' * 70}")

    print("\nSummary:")
    for name, config in thresholds.items():
        if not name.startswith("individual_signal_"):
            continue
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