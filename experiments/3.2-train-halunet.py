"""
Train HaluNet baseline on multi-year data.

This script loads training data (RealtimeQA 2022-2025), extracts the three
HaluNet input features (log-likelihoods, entropies, hidden-state embeddings),
performs a grouped train/val split, trains the HaluNet model, and saves the
checkpoint.

HaluNet requires:
- log_likelihoods: per-token log p(x_t | x_{<t})
- entropies: per-token H_t = -sum(p * log(p))
- embeddings: per-token hidden states (last layer)

These features are computed when compute_metrics(return_baseline_features=True)
is used during generation.

Usage:
    python 3.2-train-halunet.py --train-years 2022 2023 2024 2025

Output:
    models/<model_slug>/halunet.pt
    models/<model_slug>/halunet_train_summary.json
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from moeuncert.baselines import HaluNet
from moeuncert.experiments import (
    resolve_model_slug,
    resolve_dataset_slug,
    resolve_cache_dir,
    load_multi_year_data,
    stream_multi_year_data,
    stratified_group_split,
    find_generation_boundaries,
)


def extract_halunet_features(
    outputs: Dict[str, torch.Tensor],
    df_labeled: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, List[int]]:
    """Extract HaluNet features (log_likelihoods, entropies, embeddings) per answer.

    Parameters
    ----------
    outputs : Dict[str, torch.Tensor]
        Model outputs from load_multi_year_data().
    df_labeled : pd.DataFrame
        Labeled dataframe with question_id mapping.

    Returns
    -------
    Tuple of (log_likelihoods_list, entropies_list, embeddings_list, labels, valid_indices)
        Each list element corresponds to one answer (variable-length sequences).
        ``valid_indices`` is the list of positional indices into
        ``df_labeled`` for each returned answer, so the caller can filter
        ``df_labeled`` to match the extracted features.
    """
    # Check required keys exist
    required_keys = ["log_likelihoods", "entropies", "last_hidden_states", "question_id", "sequences", "input_ids"]
    missing = [k for k in required_keys if k not in outputs]
    if missing:
        raise KeyError(
            f"Missing required features in outputs: {missing}. "
            f"Ensure compute_metrics(return_baseline_features=True) was used during generation."
        )

    # Build a mapping from string question ID to a list of positional indices.
    from collections import defaultdict
    all_qids = outputs["question_id"]  # list of strings
    qid_to_indices: dict = defaultdict(list)
    for i, qid in enumerate(all_qids):
        qid_to_indices[str(qid)].append(i)
    ev_flags = outputs.get("evidence_present", [False] * len(all_qids))

    log_likelihoods_list = []
    entropies_list = []
    embeddings_list = []
    labels = []
    valid_indices = []
    skipped_no_output = 0
    skipped_empty_gen = 0

    for row_pos, (_, row) in enumerate(df_labeled.iterrows()):
        qid = str(row["question_id"])
        if qid not in qid_to_indices:
            skipped_no_output += 1
            continue

        indices = qid_to_indices[qid]
        evidence_present = bool(row.get("evidence_present", False))
        idx = indices[1] if evidence_present and len(indices) > 1 else indices[0]

        input_ids = outputs["input_ids"][idx]
        sequences = outputs["sequences"][idx]
        gen_start, gen_end = find_generation_boundaries(input_ids, sequences)
        gen_len = gen_end - gen_start

        if gen_len <= 0:
            skipped_empty_gen += 1
            continue

        ll = outputs["log_likelihoods"][idx, :gen_len].numpy()
        ent = outputs["entropies"][idx, :gen_len].numpy()
        hidden = outputs["last_hidden_states"][idx, :gen_len, :].numpy()

        # Sanitize: Gemma 4's 262K vocab can produce -inf log-likelihoods
        # (float16 log_softmax underflow for low-probability tokens).
        # Replace with finite values so HaluNet doesn't produce nan loss.
        ll = np.nan_to_num(ll, nan=0.0, posinf=0.0, neginf=-100.0)
        ent = np.nan_to_num(ent, nan=0.0, posinf=100.0, neginf=0.0)
        hidden = np.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)

        log_likelihoods_list.append(ll)
        entropies_list.append(ent)
        embeddings_list.append(hidden)
        valid_indices.append(row_pos)

        import math
        llm_val = row.get("label_llm_answer")
        if llm_val is not None and not (isinstance(llm_val, float) and math.isnan(llm_val)):
            labels.append(int(llm_val))
        elif "label_weak_hallucination" in row:
            labels.append(int(row["label_weak_hallucination"]))
        else:
            labels.append(0)

    total_skipped = skipped_no_output + skipped_empty_gen
    if total_skipped > 0:
        print(f"  Skipped {total_skipped} sample(s): "
              f"{skipped_no_output} no model output, {skipped_empty_gen} empty/zero-length generation")
        if skipped_empty_gen > len(df_labeled) * 0.05:
            print(f"  WARNING: {skipped_empty_gen} empty-generation skips (>5% of {len(df_labeled)} rows). "
                  "This may indicate a pad_token_id mismatch — check that load_tokenizer_for_data() is used.")

    return (
        log_likelihoods_list,
        entropies_list,
        embeddings_list,
        np.array(labels),
        valid_indices,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train HaluNet baseline on multi-year data"
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
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of groups for validation (default: 0.1)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save HaluNet checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available, else cpu)",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_slug = resolve_model_slug(args.model)
    save_dir = args.save_dir / model_slug
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("TRAINING HALUNET")
    print(f"{'=' * 70}")
    print(f"Train years: {args.train_years}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")

    # Load data via streaming, keeping only the keys HaluNet needs.
    # This drops expert_hidden_scores, attention_scores, and other heavy
    # keys BEFORE the per-year concat, reducing peak RAM by an order
    # of magnitude.  Combined with per-year streaming, this fixes the
    # OOM that 3.2 used to hit on combined-dir datasets.
    halunet_keys = {
        "log_likelihoods", "entropies", "last_hidden_states",
        "question_id", "sequences", "input_ids",
    }
    print("\nLoading training data (streaming, filtered to HaluNet keys)...")
    from moeuncert.experiments import load_tokenizer_for_data
    _load_tokenizer = load_tokenizer_for_data(args.model)
    per_year_dfs: List[pd.DataFrame] = []
    per_year_outputs: List[Dict] = []
    for df, outputs, _data_dir in stream_multi_year_data(
        args.data_root,
        args.train_years,
        args.month,
        args.model,
        args.label_model,
        filter_keys=halunet_keys,
        pad_token_id=_load_tokenizer.pad_token_id,
    ):
        per_year_dfs.append(df)
        per_year_outputs.append(outputs)
    df_labeled = pd.concat(per_year_dfs, ignore_index=True)
    from moeuncert.experiments.data_loading import _concat_parts
    if len(per_year_outputs) == 1:
        outputs = per_year_outputs[0]
    else:
        outputs = _concat_parts(per_year_outputs, pad_token_id=_load_tokenizer.pad_token_id)
    del per_year_dfs, per_year_outputs

    # Extract HaluNet features
    print("\nExtracting HaluNet features...")
    try:
        ll_list, ent_list, emb_list, labels, valid_indices = extract_halunet_features(
            outputs, df_labeled
        )
    except KeyError as e:
        print(f"\nERROR: {e}")
        print("\nTo generate HaluNet features, run the generation script with:")
        print("  compute_metrics(outputs, return_baseline_features=True)")
        print("\nThis computes log_likelihoods and entropies from the scores tensor.")
        raise

    print(f"  Extracted {len(ll_list)} answers")
    print(f"  Avg sequence length: {np.mean([len(x) for x in ll_list]):.1f}")
    print(f"  Hallucination rate: {labels.mean():.1%}")

    # Filter df_labeled to match the extracted features (drops rows with
    # no model output or empty generation).
    df_labeled = df_labeled.iloc[valid_indices].reset_index(drop=True)

    # Stratified train/val split
    print(f"\n{'=' * 70}")
    print("TRAIN/VAL SPLIT")
    print(f"{'=' * 70}")

    if "question_id" not in df_labeled.columns:
        raise KeyError("df_labeled must contain a 'question_id' column for grouped splitting")

    if len(df_labeled) != len(labels):
        raise ValueError(
            f"Length mismatch after filtering: {len(df_labeled)} df rows vs {len(labels)} labels"
        )

    answer_qids = df_labeled["question_id"].astype(str)

    # If the dataset includes an evidence-mode flag, include it in the grouping
    # key so variants of the same question/evidence setting are kept together.
    for evidence_col in ("evidence_present", "has_evidence", "with_evidence", "use_evidence", "rag", "use_rag"):
        if evidence_col in df_labeled.columns:
            answer_qids = answer_qids + "::" + df_labeled[evidence_col].astype(str)
            break

    train_mask, val_mask = stratified_group_split(
        labels, answer_qids.to_numpy(), test_size=args.val_fraction, random_state=args.seed
    )

    # Split features
    ll_train = [ll_list[i] for i in np.where(train_mask)[0]]
    ll_val = [ll_list[i] for i in np.where(val_mask)[0]]
    ent_train = [ent_list[i] for i in np.where(train_mask)[0]]
    ent_val = [ent_list[i] for i in np.where(val_mask)[0]]
    emb_train = [emb_list[i] for i in np.where(train_mask)[0]]
    emb_val = [emb_list[i] for i in np.where(val_mask)[0]]
    y_train = labels[train_mask]
    y_val = labels[val_mask]

    print(f"  Train answers: {len(y_train)}")
    print(f"  Val answers: {len(y_val)}")

    # Determine embedding dimension from data
    embedding_dim = emb_train[0].shape[1]
    print(f"  Embedding dimension: {embedding_dim}")

    # Train HaluNet
    print(f"\n{'=' * 70}")
    print("TRAINING HALUNET")
    print(f"{'=' * 70}")

    halunet = HaluNet(
        embedding_dim=embedding_dim,
        hidden_dim=128,
        max_seq_len=50,
        dropout=0.5,
        fusion="attention",
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"  Device: {halunet.device}")

    halunet.fit(
        ll_train,
        ent_train,
        emb_train,
        y_train,
        val_split=0.0,  # We already split manually
        verbose=True,
    )

    # Evaluate on validation set
    print(f"\n{'=' * 70}")
    print("VALIDATION EVALUATION")
    print(f"{'=' * 70}")

    val_probs = []
    for i in range(len(y_val)):
        prob = halunet.predict_proba(ll_val[i], ent_val[i], emb_val[i])
        val_probs.append(prob)
    val_probs = np.array(val_probs)

    # Sanitize: HaluNet can produce NaN probabilities if the model
    # diverged during training. Replace before passing to sklearn.
    val_probs = np.nan_to_num(val_probs, nan=0.0, posinf=1.0, neginf=0.0)

    # Compute metrics
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.metrics import precision_recall_curve
    from moeuncert.experiments import safe_roc_auc_score

    auroc = safe_roc_auc_score(y_val, val_probs)

    # Find optimal threshold for F1
    precisions, recalls, thresholds_pr = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    opt_idx = np.argmax(f1_scores)
    opt_threshold = thresholds_pr[opt_idx] if opt_idx < len(thresholds_pr) else 0.5

    val_preds = (val_probs >= opt_threshold).astype(int)
    f1 = f1_score(y_val, val_preds)
    acc = accuracy_score(y_val, val_preds)

    print(f"  Val AUROC: {auroc:.4f}")
    print(f"  Val F1 (optimal threshold={opt_threshold:.3f}): {f1:.4f}")
    print(f"  Val Accuracy: {acc:.4f}")

    # Save model
    model_path = save_dir / "halunet.pt"
    halunet.save(model_path)
    print(f"\nSaved HaluNet checkpoint to: {model_path}")

    # Save summary
    summary = {
        "model": "HaluNet",
        "train_years": args.train_years,
        "train_answers": int(len(y_train)),
        "val_answers": int(len(y_val)),
        "train_hallucination_rate": float(y_train.mean()),
        "val_hallucination_rate": float(y_val.mean()),
        "embedding_dim": embedding_dim,
        "hidden_dim": 128,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "val_auroc": float(auroc),
        "val_f1": float(f1),
        "val_accuracy": float(acc),
        "optimal_threshold": float(opt_threshold),
    }

    summary_path = save_dir / "halunet_train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to: {summary_path}")

    print(f"\n{'=' * 70}")
    print("HALUNET TRAINING COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
