"""
Data loading utilities for experiment scripts.

Provides functions for loading labeled datasets, model outputs, and multi-year
data across RealtimeQA experiments.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
            # Flatten list-of-lists: each year's outputs have a list of strings
            all_qids = []
            for o in all_outputs_list:
                all_qids.extend(o[key])
            combined_outputs[key] = all_qids
        elif isinstance(all_outputs_list[0][key], torch.Tensor):
            combined_outputs[key] = torch.cat(
                [o[key] for o in all_outputs_list], dim=0
            )
        else:
            combined_outputs[key] = all_outputs_list[0][key]

    return df_combined, combined_outputs