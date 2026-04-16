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
import inspect
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from xgboost import XGBClassifier

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets import fetch_realtimeqa
from moeuncert.experiments import resolve_model_slug, resolve_dataset_slug, read_and_collate_outputs


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


def _create_token_labels(
    generated_text: str,
    hallucinated_spans: np.ndarray,
    offset_mapping: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Create binary token labels based on hallucinated text spans.
    
    A token is labeled as hallucinated (1) if it overlaps with any 
    hallucinated span in the text.
    """
    n_tokens = len(offset_mapping)
    token_labels = torch.zeros(n_tokens, dtype=torch.long)
    
    if not isinstance(hallucinated_spans, np.ndarray) or len(hallucinated_spans) == 0:
        return token_labels
    
    for span_text in hallucinated_spans:
        span_start = generated_text.find(span_text)
        if span_start == -1:
            continue
        span_end = span_start + len(span_text)
        
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_end > span_start and char_start < span_end:
                token_labels[token_idx] = 1
    
    return token_labels


def _find_generation_boundaries(
    input_ids: torch.Tensor,
    sequences: torch.Tensor,
) -> Tuple[int, int]:
    """
    Find where the generated answer starts and ends in the sequences tensor.
    
    Compares input_ids (prompt) with sequences (prompt + generation) to find
    where they diverge, which marks the start of generation.
    
    Parameters
    ----------
    input_ids : torch.Tensor
        The prompt tokens (may include left-padding)
    sequences : torch.Tensor
        The full sequence (prompt + generation + possibly right-padding)
    
    Returns
    -------
    Tuple[int, int]
        (gen_start, gen_end) positions in sequences
    """
    # Find where actual content starts (after left-padding)
    pad_token = input_ids[0].item()  # Left-padding token (e.g., 50279)
    non_pad_mask = input_ids != pad_token
    input_content_start = (
        torch.where(non_pad_mask)[0][0].item()
        if torch.any(non_pad_mask) else len(input_ids)
    )
    
    # Find where actual content ends (before right-padding with 0)
    non_zero_mask = input_ids != 0
    input_content_end = (
        torch.where(non_zero_mask)[0][-1].item() + 1
        if torch.any(non_zero_mask) else 0
    )
    
    # Find where content starts in sequences
    seq_non_pad_mask = sequences != pad_token
    seq_content_start = (
        torch.where(seq_non_pad_mask)[0][0].item()
        if torch.any(seq_non_pad_mask) else len(sequences)
    )
    
    # Generation starts after the prompt content
    content_len = input_content_end - input_content_start
    gen_start = seq_content_start + content_len
    
    # Find where generation ends (first 0 after gen_start, or end of tensor)
    seq_zeros = torch.where(sequences[gen_start:] == 0)[0]
    gen_end = (
        gen_start + seq_zeros[0].item()
        if len(seq_zeros) > 0 else len(sequences)
    )
    
    return gen_start, gen_end


def _extract_token_features(
    all_outputs: Dict[str, torch.Tensor],
    tensor_idx: int,
    gen_start: int,
    gen_end: int,
    n_tokens: int,
) -> Dict[str, torch.Tensor]:
    """
    Extract per-token features from model outputs for the generated tokens.
    
    Hidden features are shifted by 1 (hidden[i] predicts sequences[i+1]),
    so we use gen_start-1 to gen_end-1 for hidden state features.
    """
    # Hidden features are shifted: hidden[i] predicts sequences[i+1]
    hidden_start = gen_start - 1
    hidden_end = gen_end - 1
    
    token_features = {}
    
    for key, tensor in all_outputs.items():
        if key == 'question_id':
            continue
        elif key == 'sequences':
            token_features[key] = tensor[tensor_idx, gen_start:gen_end]
        elif key in ['input_ids', 'scores_entropy']:
            # Skip - not aligned with generated tokens
            continue
        elif key == 'attention_scores':
            # (n_samples, n_heads, n_heads, seq_len) -> (n_tokens, n_heads*n_heads)
            attn = tensor[tensor_idx, :, :, hidden_start:hidden_end]
            attn = attn.permute(2, 0, 1)
            token_features[key] = attn.reshape(n_tokens, -1)
        elif key == 'expert_usage':
            # (n_samples, seq_len, n_layers, n_experts) -> (n_tokens, n_layers*n_experts)
            usage = tensor[tensor_idx, hidden_start:hidden_end]
            token_features[key] = usage.reshape(n_tokens, -1)
        elif tensor.ndim >= 2:
            token_features[key] = tensor[tensor_idx, hidden_start:hidden_end]
    
    return token_features


def prepare_training_data(
    df_labeled: pd.DataFrame,
    all_outputs: Dict[str, torch.Tensor],
    tokenizer: AutoTokenizer,
) -> Tuple[Dict, Dict]:
    """
    Prepare training data by combining per-token labels and per-token model
    internals.
    
    This function creates token-level features and labels by creating binary
    label masks (0=correct, 1=hallucinated) for each token using the provided
    hallucinated spans. It then splits the token-level features from the model
    outputs and aligns them with the token-level labels. 

    The resulting features and labels are concatenated across all samples to create
    a large token-level dataset for training detection models.
    
    Parameters
    ----------
    df_labeled : pd.DataFrame
        Labeled dataset with supervision signals including:
        - generated_answer: The text generated by the model
        - llm_hallucinated_spans: Array of hallucinated text spans
        - evidence_present: Whether evidence was provided (0 or 1)
    all_outputs : Dict[str, torch.Tensor]
        Collated outputs from base and evidence generation (no evidence + RAG)
    tokenizer : AutoTokenizer
        Tokenizer to use.
    
    Returns
    -------
    Tuple[Dict, Dict]
        (features_dict, labels_dict) where:
        - features_dict contains per-token feature tensors:
            - hidden_states: (total_tokens, n_layers, hidden_size)
            - expert_idx: (total_tokens, n_layers, top_k) 
            - expert_weights: (total_tokens, n_layers, top_k)
            - etc.
        - labels_dict contains per-token labels:
            - hallucination: (total_tokens,) binary labels
            - sample_idx: (total_tokens,) which sample each token came from
            - token_position: (total_tokens,) position within sequence
    """
    
    print("Preparing token-level training data...")
    print(f"  Labeled samples: {len(df_labeled)}")
    print(f"  Output keys: {list(all_outputs.keys())}")
    
    # Build question_id -> tensor index mapping for safe alignment
    if 'question_id' not in all_outputs:
        raise ValueError("all_outputs must contain 'question_id' for alignment")
    
    qid_to_idx = {qid.item(): idx for idx, qid in enumerate(all_outputs['question_id'])}
    
    features_list = []
    labels_list = []
    confidence_list = []
    
    for _, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="  Processing"):
        question_id = row['question_id']
        
        # Look up the tensor index using question_id
        if question_id not in qid_to_idx:
            raise KeyError(f"Question ID {question_id} not found in model outputs.")
        
        tensor_idx = qid_to_idx[question_id]
        generated_text = row['generated_answer']
        
        # Tokenize the generated answer with offset mapping for span alignment
        tokenized = tokenizer(
            generated_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        n_tokens = len(tokenized['input_ids'])
        
        # Create token-level labels from hallucinated spans
        token_labels = _create_token_labels(
            generated_text,
            row['llm_hallucinated_spans'],
            tokenized['offset_mapping'],
        )
        
        # Find where generation starts and ends in the sequences tensor
        input_ids = all_outputs['input_ids'][tensor_idx]
        sequences = all_outputs['sequences'][tensor_idx]
        gen_start, gen_end = _find_generation_boundaries(input_ids, sequences)
        
        # Use the smaller of detected generation length and tokenized length
        gen_len = gen_end - gen_start
        if gen_len < n_tokens:
            n_tokens = gen_len
            token_labels = token_labels[:n_tokens]
        else:
            gen_end = gen_start + n_tokens
        
        # Extract per-token features
        token_features = _extract_token_features(
            all_outputs, tensor_idx,
            gen_start, gen_end,
            n_tokens,
        )
        
        # Add metadata
        token_features['question_id'] = torch.full((n_tokens,), question_id, dtype=torch.long)
        token_features['token_position'] = torch.arange(n_tokens, dtype=torch.long)
        token_features['evidence_present'] = torch.full((n_tokens,), row['evidence_present'], dtype=torch.long)
        
        features_list.append(token_features)
        labels_list.append(token_labels)
        confidence_list.append(
            torch.full((n_tokens,), row['label_hallucination_confidence'], dtype=torch.float)
        )
     
    # Concatenate all samples
    print("  Concatenating all tokens...")
    features = {}
    for key in features_list[0].keys():
        features[key] = torch.cat([f[key] for f in features_list], dim=0)
    
    labels = {
        'hallucination': torch.cat(labels_list, dim=0),
    }
    labels['confidence'] = torch.cat(confidence_list, dim=0) * labels['hallucination']
    
    print(f"  Total tokens: {len(labels['hallucination'])}")
    print(f"  Hallucinated tokens: {labels['hallucination'].sum().item()} ({100 * labels['hallucination'].float().mean().item():.1f}%)")
    print("  Feature shapes:")
    for key, tensor in features.items():
        print(f"    {key}: {tensor.shape}")
    
    return features, labels


def merge_features(
    features: Dict[str, torch.Tensor],
    include_sequences: bool = False,
    include_metadata: bool = False,
) -> np.ndarray:
    """
    Merge all feature tensors into a single feature matrix.
    
    Parameters
    ----------
    features : Dict[str, torch.Tensor]
        Dictionary of feature tensors from prepare_training_data()
    include_sequences : bool, default=False
        Whether to include the sequences (token IDs) as a feature
    include_metadata : bool, default=False
        Whether to include metadata features (question_id, token_position, evidence_present)
    
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_tokens, n_features)
    """
    # Define which features to include
    numerical_features = [
        'hidden_scores',
        'attention_scores',
        'router_entropy',
        'expert_hidden_scores',
        'expert_similarities',
        'expert_usage',
    ]
    
    metadata_features = [
        'question_id',
        'token_position',
        'evidence_present',
    ]
    
    # Build list of features to merge
    features_to_merge = []
    
    # Add numerical features
    for key in numerical_features:
        if key in features:
            tensor = features[key]
            # Handle 1D tensors (add dimension)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            features_to_merge.append(tensor)
    
    # Optionally add sequences
    if include_sequences and 'sequences' in features:
        seq_tensor = features['sequences']
        if seq_tensor.ndim == 1:
            seq_tensor = seq_tensor.unsqueeze(1)
        features_to_merge.append(seq_tensor.float())
    
    # Optionally add metadata
    if include_metadata:
        for key in metadata_features:
            if key in features:
                tensor = features[key]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(1)
                features_to_merge.append(tensor.float())
    
    # Concatenate along feature dimension
    X = torch.cat(features_to_merge, dim=1)
    
    # Convert to numpy
    return X.numpy()


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
        default="glm-5.1",
        help="Label model name (e.g., 'glm-5.1' for results_labeled_glm-5.1.parquet)"
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

    # Optionally print example samples (useful for reports)
    # print_example_samples(df_labeled: pd.DataFrame, n_samples: int = 1)
    
    # Load model outputs for base generation
    print("\nLoading generated outputs...")
    base_outputs_dir = data_dir / "base_generation"
    evidence_outputs_dir = data_dir / "evidence_generation"

    # Find all batch output files
    batch_files = [
        *sorted(base_outputs_dir.glob("model_outputs__batch_*.pt")),
        *sorted(evidence_outputs_dir.glob("model_outputs__batch_*.pt")),
    ]
    
    if not batch_files:
        raise FileNotFoundError(f"No batch output files found in {output_dir}")
     
    # Use handle the loading and collation
    all_outputs = read_and_collate_outputs(batch_files, tokenizer=None, get_keys=None)

    #####################################################################################
    # Prepare training data

    # Load tokenizer for feature preparation
    tokenizer = AutoTokenizer.from_pretrained(args.model)
 
    # Prepare training data
    features, labels = prepare_training_data(
        df_labeled,
        all_outputs,
        tokenizer,
    )

    # Merge features into a single matrix
    print("\nMerging features...")
    X = merge_features(
        features,
        include_sequences=False,  # Token IDs not useful for most models
        include_metadata=False,   # Keep metadata separate for analysis
    )
    y = labels['hallucination'].numpy()
    y_confidence = labels['confidence'].numpy()
    
    # Use question_id for train/val split to avoid leakage
    question_ids = features['question_id'].numpy()

    # Grouped train/test split by question_id (prevents token leakage across splits)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=question_ids))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    question_ids_train = question_ids[train_idx]
    question_ids_test = question_ids[test_idx]
    
    y_confidence_train = y_confidence[train_idx]
    group_kfold = GroupKFold(n_splits=5)

    _clean_inf = FunctionTransformer(lambda X: np.where(np.isfinite(X), X, np.nan))
    _imputer = SimpleImputer(strategy="median")

    model_configs = {
        "LogisticRegression": {
            "pipeline": Pipeline([
                ("clean_inf", _clean_inf),
                ("imputer", _imputer),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=args.seed, l1_ratio=0.0)),
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("clean_inf", _clean_inf),
                ("imputer", _imputer),
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=args.seed, n_jobs=-1)),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [10, 20, None],
                "clf__min_samples_leaf": [1, 5],
            },
        },
        "XGBoost": {
            "pipeline": Pipeline([
                ("clean_inf", _clean_inf),
                ("imputer", _imputer),
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    random_state=args.seed,
                    n_jobs=-1,
                    eval_metric="logloss",
                    verbosity=0,
                )),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [3, 6, 10],
                "clf__learning_rate": [0.01, 0.1, 0.3],
            },
        },
        "MLP": {
            "pipeline": Pipeline([
                ("clean_inf", _clean_inf),
                ("imputer", _imputer),
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(random_state=args.seed, early_stopping=True)),
            ]),
            "param_grid": {
                "clf__hidden_layer_sizes": [(128,), (256, 128), (128, 64)],
                "clf__alpha": [1e-4, 1e-3, 1e-2],
                "clf__learning_rate_init": [1e-3, 1e-4],
            },
        },
    }

    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION (target: y, grouped k-fold CV)")
    print("=" * 70)

    binary_results = {}
    for name, config in model_configs.items():
        print(f"\n--- {name} ---")
        search = GridSearchCV(
            config["pipeline"],
            config["param_grid"],
            cv=group_kfold.split(X_train, y_train, groups=question_ids_train),
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred_proba = best.predict_proba(X_test)[:, 1]
        y_pred = best.predict(X_test)
        fscore = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred_proba)
        print(f"  Best params: {search.best_params_}")
        print(f"  Best CV F1: {search.best_score_:.4f}")
        print(f"  Test F1: {fscore:.4f}")
        print(f"  Test AP: {ap:.4f}")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        binary_results[name] = {
            "model": best,
            "fscore": fscore,
            "ap": ap,
            "y_pred_proba": y_pred_proba,
        }

    print("\n" + "=" * 70)
    print("CONFIDENCE-WEIGHTED CLASSIFICATION (sample_weight from hallucination confidence)")
    print("=" * 70)

    sw_train = np.where(y_train == 1, y_confidence_train, 1.0 - y_confidence_train)
    sw_train = np.clip(sw_train, 1e-6, None)

    weighted_results = {}
    for name, config in model_configs.items():
        print(f"\n--- {name} (confidence-weighted) ---")

        pipeline = config["pipeline"]
        supports_sample_weight = hasattr(pipeline.named_steps["clf"], "fit")
        if supports_sample_weight:
            fit_params = inspect.signature(pipeline.named_steps["clf"].fit).parameters
            supports_sample_weight = "sample_weight" in fit_params

        search = GridSearchCV(
            pipeline,
            config["param_grid"],
            cv=group_kfold.split(X_train, y_train, groups=question_ids_train),
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )

        if supports_sample_weight:
            search.fit(X_train, y_train, clf__sample_weight=sw_train)
        else:
            print("  (model does not support sample_weight; training unweighted)")
            search.fit(X_train, y_train)

        best = search.best_estimator_
        y_pred_proba = best.predict_proba(X_test)[:, 1]
        y_pred = best.predict(X_test)
        fscore = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred_proba)
        print(f"  Best params: {search.best_params_}")
        print(f"  Best CV F1: {search.best_score_:.4f}")
        print(f"  Test F1: {fscore:.4f}")
        print(f"  Test AP: {ap:.4f}")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        weighted_results[name] = {
            "model": best,
            "fscore": fscore,
            "ap": ap,
            "y_pred_proba": y_pred_proba,
        }

    print("\n" + "=" * 70)
    print("SUMMARY: Binary vs Confidence-Weighted")
    print("=" * 70)
    print(f"{'Model':<20} {'Binary F1':>12} {'Weighted F1':>14} {'Binary AP':>12} {'Weighted AP':>14}")
    print("-" * 72)
    for name in model_configs:
        b = binary_results[name]
        w = weighted_results[name]
        print(f"{name:<20} {b['fscore']:>12.4f} {w['fscore']:>14.4f} {b['ap']:>12.4f} {w['ap']:>14.4f}")

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label vector shape: {y.shape}")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Train question_ids: {len(np.unique(question_ids_train))}")
    print(f"  Test question_ids: {len(np.unique(question_ids_test))}")
    print(f"  Total features: {X.shape[1]}")
    print("    - hidden_scores: 17")
    print("    - attention_scores: 256")
    print("    - router_entropy: 16")
    print("    - expert_hidden_scores: 16")
    print("    - expert_similarities: 16")
    print("    - expert_usage: 1024")
    print(f"  Hallucination rate: {y.mean():.1%}")
 
