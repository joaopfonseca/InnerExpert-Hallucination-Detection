"""
Train hallucination detection models using labeled data and model internals.

This script loads:
1. Labeled dataset from 2.0-make-labels.py (parquet with supervision signals)
2. Raw model outputs from 1.0-generate-answers.py (.pt files with hidden states, expert routing)

It then trains detection models using:
- Token-level features: hidden states, expert routing patterns
- Sequence-level features: aggregated uncertainty metrics
- Answer-level labels: hallucination confidence scores

The script supports multiple detection architectures and evaluation metrics.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets import fetch_realtimeqa
from moeuncert.experiments import resolve_model_slug, resolve_dataset_slug

def load_labeled_dataset(
    model_dir: Path,
    label_model: str = None,
) -> pd.DataFrame:
    """
    Load the labeled parquet file from 2.0-make-labels.py.
    
    Parameters
    ----------
    data_root : Path
        Root directory containing datasets
    dataset_slug : str
        Dataset identifier (e.g., 'realtimeqa-2026-02')
    model_slug : str
        Model identifier with '/' replaced by '__'
    label_model : str, optional
        If specified, loads results_labeled_{label_model}.parquet
        Otherwise loads results.parquet
    
    Returns
    -------
    pd.DataFrame
        Labeled dataset with columns including:
        - question_sentence, answer, generated_answer
        - rouge1, rouge2, rougeL, bert_f1, etc.
        - label_hallucination_confidence
        - label_llm_answer (if available)
        - llm_hallucinated_spans (if available)
    """
    
    if label_model:
        parquet_path = model_dir / f"results_labeled_{label_model}.parquet"
    else:
        parquet_path = model_dir / "results.parquet"
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"Loaded labeled dataset: {parquet_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    return df


def load_model_outputs(
    output_dir: Path,
) -> List[Dict]:
    """
    Load raw model outputs (.pt files) from 1.0-generate-answers.py.
    
    Parameters
    ----------
    data_root : Path
        Root directory containing datasets
    dataset_slug : str
        Dataset identifier
    model_slug : str
        Model identifier
    
    Returns
    -------
    List[Dict]
        List of batch outputs, each containing:
        - input_ids: (batch_size, seq_len)
        - hidden_states: (batch_size, seq_len, n_layers, hidden_size)
        - expert_idx: (batch_size, seq_len, n_layers, top_k)
        - expert_weights: (batch_size, seq_len, n_layers, top_k)
        - generated_ids: (batch_size, max_new_tokens)
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"Model outputs directory not found: {output_dir}")
    
    # Find all batch output files
    batch_files = sorted(output_dir.glob("model_outputs__batch_*.pt"))
    
    if not batch_files:
        raise FileNotFoundError(f"No batch output files found in {output_dir}")
    
    print(f"Loading {len(batch_files)} batch files from {output_dir}")
    
    all_outputs = []
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        batch_data = torch.load(batch_file, map_location="cpu")
        all_outputs.append(batch_data)
    
    print(f"  Loaded {len(all_outputs)} batches")
    
    return all_outputs


def collate_batches(batch_outputs: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch outputs into single tensors.
    
    Parameters
    ----------
    batch_outputs : List[Dict]
        List of batch outputs from load_model_outputs
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Collated tensors with batch dimension concatenated:
        - hidden_states: (total_samples, seq_len, n_layers, hidden_size)
        - expert_idx: (total_samples, seq_len, n_layers, top_k)
        - expert_weights: (total_samples, seq_len, n_layers, top_k)
        - generated_ids: (total_samples, max_new_tokens)
    """
    # Determine which keys to collate
    sample_batch = batch_outputs[0]
    keys_to_collate = [k for k in sample_batch.keys() if isinstance(sample_batch[k], torch.Tensor)]
    
    collated = {}
    
    for key in keys_to_collate:
        tensors = [batch[key] for batch in batch_outputs]
        
        # Check if all tensors have the same shape (except batch dimension)
        shapes = [t.shape[1:] for t in tensors]
        if len(set(shapes)) > 1:
            print(f"Warning: {key} has varying shapes across batches, padding...")
            # Pad to max shape
            max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
            padded_tensors = []
            for t in tensors:
                if t.shape[1:] != max_shape:
                    # Create padded tensor
                    pad_shape = (t.shape[0],) + max_shape
                    padded = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                    # Copy original data
                    slices = (slice(None),) + tuple(slice(0, s) for s in t.shape[1:])
                    padded[slices] = t
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(t)
            tensors = padded_tensors
        
        collated[key] = torch.cat(tensors, dim=0)
    
    print(f"Collated tensors:")
    for key, tensor in collated.items():
        print(f"  {key}: {tensor.shape}")
    
    return collated


def prepare_training_data(
    labeled_df: pd.DataFrame,
    base_outputs: Dict[str, torch.Tensor],
    evidence_outputs: Dict[str, torch.Tensor],
) -> Tuple[Dict, Dict]:
    """
    Prepare training data by combining labels and model internals.
    
    Parameters
    ----------
    labeled_df : pd.DataFrame
        Labeled dataset with supervision signals
    base_outputs : Dict[str, torch.Tensor]
        Collated outputs from base generation (no evidence)
    evidence_outputs : Dict[str, torch.Tensor]
        Collated outputs from evidence generation (RAG)
    
    Returns
    -------
    Tuple[Dict, Dict]
        (features_dict, labels_dict) where:
        - features_dict contains all feature tensors
        - labels_dict contains all supervision signals
    """
    # TODO: Implement feature extraction and alignment
    
    # For now, just return the raw data
    features = {
        "base_hidden_states": base_outputs.get("hidden_states"),
        "base_expert_idx": base_outputs.get("expert_idx"),
        "base_expert_weights": base_outputs.get("expert_weights"),
        "evidence_hidden_states": evidence_outputs.get("hidden_states"),
        "evidence_expert_idx": evidence_outputs.get("expert_idx"),
        "evidence_expert_weights": evidence_outputs.get("expert_weights"),
    }
    
    labels = {
        "hallucination_confidence": labeled_df["label_hallucination_confidence"].values,
        "weak_hallucination": labeled_df.get("label_weak_hallucination", pd.Series([None] * len(labeled_df))).values,
    }
    
    # Include LLM labels if available
    if "label_llm_answer" in labeled_df.columns:
        labels["llm_answer"] = labeled_df["label_llm_answer"].values
    
    print(f"Prepared training data:")
    print(f"  Features: {list(features.keys())}")
    print(f"  Labels: {list(labels.keys())}")
    print(f"  Samples: {len(labeled_df)}")
    
    return features, labels

def print_example_samples(df_labeled: pd.DataFrame, n_samples: int = 1, random_state: int = 42):
    df_examples = (
        df_labeled
        .groupby(["evidence_present", "label_weak_hallucination"])
        .sample(1, random_state=random_state)
    )

    for i, (idx, example) in enumerate(df_examples.iterrows()):
        print("===================================")
        print(
            f"EXAMPLE {i+1}:", 
            f"has evidence: {example['evidence_present']}", 
            f"| is hallucination: {example['label_weak_hallucination']}"
        )
        print("===================================")
        print("Question:", example["question_sentence"])
        print("Evidence:", example["evidence"])
        print("Answer:", example["generated_answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train hallucination detection models using labeled data and model internals"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        choices=range(1, 13),
        metavar="MONTH",
        help="Month (1-12). Only valid when a single year is provided.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Year(s) for RealtimeQA, e.g. --years 2025 2026. Defaults to previous month year.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name (default: allenai/OLMoE-1B-7B-0924-Instruct)",
    )
    parser.add_argument(
        "--label-model",
        type=str,
        default="gemma3-1b",
        help="Label model name (e.g., 'gemma3-1b' for results_labeled_gemma3-1b.parquet)"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
     
    args = parser.parse_args()

    # Set random seeds
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    
    # Resolve paths
    years, month, dataset_slug = resolve_dataset_slug(args.years, args.month)
    model_slug = resolve_model_slug(args.model)

    data_dir = args.data_root / dataset_slug / model_slug

    
    #####################################################################################
    # Load data 

    # Load labeled dataset
    df_labeled = load_labeled_dataset(
        data_dir,
        args.label_model,
    )

    raise RuntimeError("Checkpoint.")

    # Optionally print example samples (useful for reports)
    # print_example_samples(df_labeled: pd.DataFrame, n_samples: int = 1)
    
    # Load model outputs for base generation
    print("\nLoading base generation outputs...")
    base_batches = load_model_outputs(
        data_dir / "base_generation"
    )
    base_outputs = collate_batches(base_batches)
    
    # Load model outputs for evidence generation
    print("\nLoading evidence generation outputs...")
    evidence_batches = load_model_outputs(
        data_dir / "evidence_generation"
    )
    evidence_outputs = collate_batches(evidence_batches)
    
    print("\n" + "=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)
    
    # Prepare training data
    features, labels = prepare_training_data(
        labeled_df,
        base_outputs,
        evidence_outputs,
    )
    
    print("\n" + "=" * 80)
    print("DATA LOADING COMPLETE")
    print("=" * 80)
    print(f"Ready to train {args.detector_type} detector")
    print(f"Features: {list(features.keys())}")
    print(f"Labels: {list(labels.keys())}")
    print(f"Total samples: {len(labeled_df)}")
    print(f"Training samples: {int(len(labeled_df) * args.train_split)}")
    print(f"Validation samples: {len(labeled_df) - int(len(labeled_df) * args.train_split)}")
    
    # TODO: Implement model training
    print("\n[TODO] Model training not yet implemented")
