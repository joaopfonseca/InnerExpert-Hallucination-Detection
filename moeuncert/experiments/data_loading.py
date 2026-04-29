"""
Data loading utilities for experiment scripts.

Provides functions for loading labeled datasets, model outputs, and multi-year
data across RealtimeQA experiments.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from .paths import resolve_model_slug, resolve_dataset_slug


def load_labeled_dataset(
    data_dir: Path,
    label_model: Optional[str] = None,
) -> pd.DataFrame:
    """Load labeled parquet from 2.0-make-labels.py.

    Parameters
    ----------
    data_dir : Path
        Directory containing results (e.g., data/realtimeqa-2025-02/OLMoE-1B-7B-0924-Instruct/)
    label_model : str, optional
        If specified, loads results_labeled_{label_model}.parquet.
        Otherwise loads results_labeled.parquet.

    Returns
    -------
    pd.DataFrame
        Labeled dataset with hallucination labels.
    """
    if label_model:
        parquet_path = data_dir / f"results_labeled_{resolve_model_slug(label_model)}.parquet"
    else:
        parquet_path = data_dir / "results_labeled.parquet"

    if not parquet_path.exists():
        # Fallback to unlabeled results.parquet (for backward compatibility)
        parquet_path = data_dir / "results.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Labeled dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"  Loaded: {parquet_path} ({len(df)} rows)")
    return df


def load_model_outputs(data_dir: Path) -> Dict[str, torch.Tensor]:
    """Load and collate model outputs from .pt batch files.

    Parameters
    ----------
    data_dir : Path
        Directory containing base_generation/ and evidence_generation/ subdirs.

    Returns
    -------
    Dict[str, torch.Tensor]
        Collated outputs from all batches.
    """
    from .utils import read_and_collate_outputs

    base_dir = data_dir / "base_generation"
    evidence_dir = data_dir / "evidence_generation"

    batch_files = []
    for d in [base_dir, evidence_dir]:
        if d.exists():
            batch_files.extend(sorted(d.glob("model_outputs__batch_*.pt")))

    if not batch_files:
        raise FileNotFoundError(f"No batch output files found in {data_dir}")

    return read_and_collate_outputs(batch_files, tokenizer=None, get_keys=None)


def load_multi_year_data(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    label_model: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
    """Load labeled data and model outputs across multiple years.

    Parameters
    ----------
    data_root : Path
        Root data directory.
    years : List[int]
        Years to load (e.g., [2022, 2023, 2024, 2025]).
    month : int, optional
        Month for single-year datasets (None = use all months per year).
    model : str
        Model name/slug.
    label_model : str, optional
        Label model name for labeled parquet filenames.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, torch.Tensor]]
        Concatenated labeled dataframe and collated model outputs.
    """
    model_slug = resolve_model_slug(model)
    all_dfs = []
    all_outputs_list = []

    print(f"\nLoading data for years: {years}")
    for year in years:
        _, _, dataset_slug = resolve_dataset_slug([year], month)
        data_dir = data_root / dataset_slug / model_slug

        if not data_dir.exists():
            print(f"  WARNING: {data_dir} not found, skipping year {year}")
            continue

        try:
            df = load_labeled_dataset(data_dir, label_model)
            df["year"] = year
            all_dfs.append(df)

            outputs = load_model_outputs(data_dir)
            all_outputs_list.append(outputs)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

    if not all_dfs:
        raise ValueError(f"No data found for years {years}")

    # Concatenate dataframes
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined labeled data: {len(df_combined)} rows")

    # Concatenate outputs (question IDs are already unique across years)
    combined_outputs = {}
    for key in all_outputs_list[0].keys():
        if key == "question_id":
            # Flatten any nested lists: batch files may store question_id as
            # List[List[str]] so each extend() call could introduce a sub-list.
            all_qids: List[str] = []
            for o in all_outputs_list:
                for item in o[key]:
                    if isinstance(item, list):
                        all_qids.extend(item)
                    else:
                        all_qids.append(item)
            combined_outputs[key] = all_qids
        elif isinstance(all_outputs_list[0][key], torch.Tensor):
            combined_outputs[key] = torch.cat(
                [o[key] for o in all_outputs_list], dim=0
            )
        else:
            combined_outputs[key] = all_outputs_list[0][key]

    # Sanity-check: question_id length must match the first tensor batch dimension.
    if "question_id" in combined_outputs:
        first_tensor_key = next(
            (k for k in combined_outputs if isinstance(combined_outputs[k], torch.Tensor)),
            None,
        )
        if first_tensor_key is not None:
            n_tensor = combined_outputs[first_tensor_key].shape[0]
            n_qids = len(combined_outputs["question_id"])
            assert n_qids == n_tensor, (
                f"question_id length ({n_qids}) does not match "
                f"tensor batch size ({n_tensor})"
            )

    return df_combined, combined_outputs


def load_sampled_outputs(
    data_root: Path,
    years: List[int],
    month: Optional[int],
    model: str,
    num_samples: int = 5,
) -> Dict[str, List]:
    """Load sampled generation outputs across multiple years.

    Reads .pt batch files from sampled_generation/ directories and collates
    them into per-question data structures needed by SemanticUncertainty and
    SemanticEnergy baselines.

    Parameters
    ----------
    data_root : Path
        Root data directory.
    years : List[int]
        Years to load.
    month : int, optional
        Month for single-year datasets.
    model : str
        Model name/slug.
    num_samples : int
        Number of sampled responses per question (default: 5).

    Returns
    -------
    Dict with keys:
        responses_by_qid : Dict[str, List[str]]
            Per-question list of sampled response strings.
        log_probs_by_qid : Dict[str, List[List[float]]]
            Per-question per-response per-token log probabilities.
        logits_by_qid : Dict[str, List[List[float]]]
            Per-question per-response per-token logits (for SemanticEnergy).
        labels_by_qid : Dict[str, int]
            Per-question hallucination label (from the base labeled dataset).
    """
    from .utils import read_and_collate_outputs

    model_slug = resolve_model_slug(model)
    all_responses_by_qid: Dict[str, List[str]] = {}
    all_logprobs_by_qid: Dict[str, List[List[float]]] = {}
    all_logits_by_qid: Dict[str, List[List[float]]] = {}

    print(f"\nLoading sampled outputs for years: {years}")
    for year in years:
        _, _, dataset_slug = resolve_dataset_slug([year], month)
        sampled_dir = data_root / dataset_slug / model_slug / "sampled_generation"

        if not sampled_dir.exists():
            print(f"  WARNING: {sampled_dir} not found, skipping year {year}")
            continue

        batch_files = sorted(sampled_dir.glob("sampled_outputs__batch_*.pt"))
        if not batch_files:
            print(f"  WARNING: No batch files in {sampled_dir}")
            continue

        # Read and collate all batches
        outputs = read_and_collate_outputs(
            batch_files, tokenizer=None, get_keys=None
        )
        print(f"  Loaded {len(batch_files)} batch files from year {year}")

        # Extract question_ids (list of strings)
        qids = outputs["question_id"]

        # For each question, collect sampled responses and log probs
        for idx, qid in enumerate(qids):
            qid = str(qid)

            # Collect responses from each sample
            responses = []
            log_probs = []
            logits = []

            for s in range(num_samples):
                seq_key = f"sequences_sample{s}"
                ll_key = f"log_likelihoods_sample{s}"
                scores_key = f"scores_sample{s}"

                if seq_key not in outputs or ll_key not in outputs:
                    continue

                # Decode the generated sequence (outputs include full seq,
                # we need only the generated part)
                gen_tokens = outputs[seq_key][idx]
                # Filter out padding (token_id == 0 or pad_token)
                gen_tokens = gen_tokens[gen_tokens != 0]

                # Get the response text — we need tokenizer for this,
                # so store token IDs and convert later, or skip text for now.
                # Store token IDs temporarily; calling code will decode if needed.
                response_tokens = gen_tokens.tolist()
                responses.append(response_tokens)

                # Log-likelihoods: (gen_seq_len,) per response
                ll = outputs[ll_key][idx].tolist()
                log_probs.append(ll)

                # Raw scores/logits: (gen_seq_len, vocab_size) per response
                if scores_key in outputs:
                    scores = outputs[scores_key][idx]  # (gen_seq_len, vocab_size)
                    # Extract logits for the generated tokens
                    gen_len = min(len(ll), scores.shape[0])
                    if gen_len > 0 and len(gen_tokens) > 0:
                        # Get logits at each position for the actual generated token
                        token_ids = gen_tokens[:gen_len]
                        response_logits = scores[:gen_len].gather(
                            -1, token_ids.unsqueeze(-1).to(scores.device)
                        ).squeeze(-1).tolist()
                        logits.append(response_logits)

            if responses:
                all_responses_by_qid[qid] = responses
                all_logprobs_by_qid[qid] = log_probs
                if logits:
                    all_logits_by_qid[qid] = logits

    print(f"  Total questions with sampled data: {len(all_responses_by_qid)}")
    return {
        "responses_by_qid": all_responses_by_qid,
        "log_probs_by_qid": all_logprobs_by_qid,
        "logits_by_qid": all_logits_by_qid,
    }