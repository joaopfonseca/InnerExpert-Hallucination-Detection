"""
Train hallucination detection models using labeled data and model internals.

This script loads:
1. Labeled datasets from 2.0-make-labels.py (parquet with supervision signals)
   across multiple years (e.g., RealtimeQA 2022-2025)
2. Raw model outputs from 1.0-generate-answers.py (.pt files with hidden states,
   expert routing, etc.) for each year

It then:
- Concatenates multi-year data into a single training corpus
- Performs stratified train/val split grouped by question_id
- Trains detection models with feature normalization
- Saves the best model and validation results

Usage:
    python 3.0-detection-model-training.py --train-years 2022 2023 2024 2025

The test evaluation (OOD on 2026) is performed by 4.0-model-evaluation.py.
"""

import argparse
import inspect
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedShuffleSplit,
)
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

from moeuncert.experiments import (
    resolve_model_slug,
    resolve_dataset_slug,
    resolve_cache_dir,
    read_and_collate_outputs,
    load_multi_year_data,
    stratified_group_split,
    compute_metrics_at_threshold,
    create_token_labels,
    find_generation_boundaries,
    replace_inf_with_nan,
)


# ---------------------------------------------------------------------------
# Transformer model (optional — requires skorch)
# ---------------------------------------------------------------------------

try:
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping
    from moeuncert.models import LayerGroupTransformerClassifier
    _HAS_SKORCH = True
except ImportError:
    _HAS_SKORCH = False


def _build_transformer_config(group_sizes):
    """Build a skorch NeuralNetClassifier for the Transformer model.

    Returns a config dict with 'model' and 'param_grid' keys, or raises
    if skorch is not installed.
    """
    if not _HAS_SKORCH:
        raise ImportError(
            "skorch is required for the Transformer model. "
            "Install with: pip install skorch"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetClassifier(
        module=LayerGroupTransformerClassifier,
        module__group_sizes=group_sizes,
        module__d_model=64,
        module__n_heads=4,
        module__n_transformer_layers=2,
        module__dropout=0.1,
        module__n_classes=2,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        max_epochs=50,
        batch_size=256,
        iterator_train__shuffle=True,
        callbacks=[EarlyStopping(patience=5, monitor="valid_loss")],
        device=device,
        verbose=0,
    )
    return model


def build_feature_pipeline(
    feature_names: List[str],
    scale_features: List[str],
    model,
    param_grid: Dict,
) -> Tuple[Pipeline, Dict]:
    """Build sklearn pipeline with selective feature normalization.

    StandardScaler is applied to all features EXCEPT expert_usage ratios
    (which are already bounded and should not be scaled).

    Parameters
    ----------
    feature_names : List[str]
        Names of all features in order.
    scale_features : List[str]
        Feature names to scale (all except expert_usage).
    model
        Sklearn estimator.
    param_grid : Dict
        Hyperparameter grid for GridSearchCV.

    Returns
    -------
    Tuple[Pipeline, Dict]
        Pipeline and param_grid for GridSearchCV.
    """
    # Map feature names to column indices
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    scale_indices = [feature_to_idx[f] for f in scale_features if f in feature_to_idx]
    pass_indices = [
        i for i, name in enumerate(feature_names)
        if name not in scale_features
    ]

    preprocessor = ColumnTransformer([
        ("scale", StandardScaler(), scale_indices),
        ("pass", "passthrough", pass_indices),
    ])

    _clean_inf = FunctionTransformer(replace_inf_with_nan)
    _imputer = SimpleImputer(strategy="median")

    pipeline = Pipeline([
        ("clean_inf", _clean_inf),
        ("imputer", _imputer),
        ("preprocessor", preprocessor),
        ("clf", model),
    ])

    return pipeline, param_grid


# =============================================================================
# Existing functions (unchanged from original)
# =============================================================================

def _extract_token_features(
    all_outputs: Dict[str, torch.Tensor],
    tensor_idx: int,
    gen_start: int,
    gen_end: int,
    n_tokens: int,
) -> Dict[str, torch.Tensor]:
    """Extract per-token features for generated tokens.

    Returns an empty dict if ``gen_end <= gen_start`` or ``n_tokens <= 0``;
    callers should skip such samples (e.g. an empty generation).
    """
    if gen_end <= gen_start or n_tokens <= 0:
        return {}
    hidden_start = gen_start - 1
    hidden_end = gen_end - 1

    token_features = {}

    for key, tensor in all_outputs.items():
        if key == 'question_id':
            continue
        elif key == 'sequences':
            token_features[key] = tensor[tensor_idx, gen_start:gen_end]
        elif key in ['input_ids', 'scores_entropy']:
            continue
        elif not isinstance(tensor, torch.Tensor):
            continue
        elif key == 'attention_scores':
            attn = tensor[tensor_idx, :, :, hidden_start:hidden_end]
            attn = attn.permute(2, 0, 1)
            token_features[key] = attn.reshape(n_tokens, -1)
        elif key == 'expert_usage':
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
    """Prepare token-level training data."""
    print("Preparing token-level training data...")
    print(f"  Labeled samples: {len(df_labeled)}")

    if 'question_id' not in all_outputs:
        raise ValueError("all_outputs must contain 'question_id' for alignment")

    # Build a mapping from string question ID to a list of positional indices.
    # The same question_id can appear twice when both base and evidence outputs
    # are present; duplicates must be disambiguated via evidence_present.
    from collections import defaultdict
    all_qids = all_outputs['question_id']  # list of strings
    qid_to_indices: dict = defaultdict(list)
    for i, qid in enumerate(all_qids):
        qid_to_indices[str(qid)].append(i)

    features_list = []
    labels_list = []
    confidence_list = []
    skipped_samples = 0

    for _, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="  Processing"):
        question_id = str(row['question_id'])

        if question_id not in qid_to_indices:
            raise KeyError(f"Question ID {question_id} not found in model outputs.")

        indices = qid_to_indices[question_id]
        # Convention: base outputs (evidence_present=False) occupy the first
        # occurrence; RAG/evidence outputs occupy the second occurrence.
        evidence_present = bool(row.get('evidence_present', False))
        tensor_idx = indices[1] if evidence_present and len(indices) > 1 else indices[0]
        generated_text = row['generated_answer']

        tokenized = tokenizer(
            generated_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        n_tokens = len(tokenized['input_ids'])

        token_labels = create_token_labels(
            generated_text,
            row['llm_hallucinated_spans'],
            tokenized['offset_mapping'],
        )

        input_ids = all_outputs['input_ids'][tensor_idx]
        sequences = all_outputs['sequences'][tensor_idx]
        gen_start, gen_end = find_generation_boundaries(input_ids, sequences)

        gen_len = gen_end - gen_start
        if gen_len <= 0:
            skipped_samples += 1
            continue
        if gen_len < n_tokens:
            n_tokens = gen_len
            token_labels = token_labels[:n_tokens]
        else:
            gen_end = gen_start + n_tokens

        if n_tokens <= 0:
            skipped_samples += 1
            continue

        token_features = _extract_token_features(
            all_outputs, tensor_idx,
            gen_start, gen_end,
            n_tokens,
        )

        token_features['question_id'] = torch.full((n_tokens,), tensor_idx, dtype=torch.long)
        token_features['token_position'] = torch.arange(n_tokens, dtype=torch.long)
        token_features['evidence_present'] = torch.full(
            (n_tokens,), row['evidence_present'], dtype=torch.long
        )

        features_list.append(token_features)
        labels_list.append(token_labels)
        confidence_list.append(
            torch.full((n_tokens,), row['label_hallucination_confidence'], dtype=torch.float)
        )

    if skipped_samples:
        print(f"  Skipped {skipped_samples} sample(s) with empty/zero-length generation")

    if not features_list:
        raise ValueError(
            f"No usable samples in labeled data "
            f"(skipped {skipped_samples}/{len(df_labeled)}). "
            "Check that model outputs and labeled dataset are aligned."
        )

    features = {}
    for key in features_list[0].keys():
        features[key] = torch.cat([f[key] for f in features_list], dim=0)

    labels = {
        'hallucination': torch.cat(labels_list, dim=0),
    }
    labels['confidence'] = torch.cat(confidence_list, dim=0) * labels['hallucination']

    print(f"  Total tokens: {len(labels['hallucination'])}")
    print(f"  Hallucinated tokens: {labels['hallucination'].sum().item()} "
          f"({100 * labels['hallucination'].float().mean().item():.1f}%)")

    return features, labels


def merge_features(
    features: Dict[str, torch.Tensor],
    include_sequences: bool = False,
    include_metadata: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Merge feature tensors into a single matrix and return feature names.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Feature matrix (n_tokens, n_features) and list of feature names.
    """
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

    features_to_merge = []
    feature_names = []

    for key in numerical_features:
        if key in features:
            tensor = features[key]
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            features_to_merge.append(tensor)
            # Add feature names
            for i in range(tensor.shape[1]):
                feature_names.append(f"{key}_{i}")

    if include_sequences and 'sequences' in features:
        seq_tensor = features['sequences']
        if seq_tensor.ndim == 1:
            seq_tensor = seq_tensor.unsqueeze(1)
        features_to_merge.append(seq_tensor.float())
        for i in range(seq_tensor.shape[1]):
            feature_names.append(f"sequence_{i}")

    if include_metadata:
        for key in metadata_features:
            if key in features:
                tensor = features[key]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(1)
                features_to_merge.append(tensor.float())
                for i in range(tensor.shape[1]):
                    feature_names.append(f"{key}_{i}")

    X = torch.cat(features_to_merge, dim=1)
    return X.numpy(), feature_names


def print_example_samples(df_labeled: pd.DataFrame, n_samples: int = 1, random_state: int = 42):
    df_examples = (
        df_labeled
        .groupby(["evidence_present", "label_weak_hallucination"])
        .sample(1, random_state=random_state)
    )

    for i, (idx, example) in enumerate(df_examples.iterrows()):
        print("=" * 35)
        print(
            f"EXAMPLE {i+1}:",
            f"has evidence: {example['evidence_present']}",
            f"| is hallucination: {example['label_weak_hallucination']}"
        )
        print("=" * 35)
        print("Question:", example["question_sentence"])
        print("Evidence:", example["evidence"])
        print("Answer:", example["generated_answer"])


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train hallucination detection models on multi-year data"
    )

    # Data arguments
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
        help="Label model name for labeled parquet filenames",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Output arguments
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained models and results",
    )

    args = parser.parse_args()

    # Set random seeds
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Create save directory
    model_slug = resolve_model_slug(args.model)
    save_dir = args.save_dir / model_slug
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")

    # =========================================================================
    # Load multi-year data
    # =========================================================================

    from moeuncert.experiments import load_tokenizer_for_data
    tokenizer = load_tokenizer_for_data(args.model)

    df_labeled, all_outputs, _ = load_multi_year_data(
        args.data_root,
        args.train_years,
        args.month,
        args.model,
        args.label_model,
        pad_token_id=tokenizer.pad_token_id,
    )

    # =========================================================================
    # Prepare token-level features and labels
    # =========================================================================

    features, labels = prepare_training_data(
        df_labeled,
        all_outputs,
        tokenizer,
    )

    # Compute per-group sizes for the Transformer model (before merge_features
    # flattens the structured tensors into a single 2D matrix).
    from moeuncert.models import compute_group_sizes

    # router_entropy covers exactly the MoE decoder layers (no embedding
    # layer), so it gives the correct layer count for the Transformer's
    # token grouping.  hidden_scores includes the embedding layer output
    # at position 0, giving n_layers+1 entries — trim it to match.
    n_layers_model = features["router_entropy"].shape[1]
    if features["hidden_scores"].shape[1] > n_layers_model:
        features["hidden_scores"] = features["hidden_scores"][:, -n_layers_model:]

    transformer_group_sizes = compute_group_sizes(features, n_layers_model)
    print(f"  Transformer group_sizes: {transformer_group_sizes}")

    # Merge features
    X, feature_names = merge_features(
        features,
        include_sequences=False,
        include_metadata=False,
    )
    y = labels['hallucination'].numpy()
    y_confidence = labels['confidence'].numpy()
    question_ids = features['question_id'].numpy()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Hallucination rate: {y.mean():.1%}")

    # =========================================================================
    # Stratified train/val split (grouped by question_id)
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("STRATIFIED TRAIN/VAL SPLIT (grouped by question_id)")
    print(f"{'=' * 70}")

    train_mask, val_mask = stratified_group_split(
        y, question_ids, test_size=args.val_fraction, random_state=args.seed
    )

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    qids_train = question_ids[train_mask]
    qids_val = question_ids[val_mask]
    y_confidence_train = y_confidence[train_mask]

    print(f"  Train tokens: {len(y_train)}")
    print(f"  Val tokens: {len(y_val)}")

    # =========================================================================
    # Feature normalization: scale all except expert_usage
    # =========================================================================

    # Identify which features are expert_usage (should not be scaled)
    scale_features = [f for f in feature_names if not f.startswith("expert_usage")]

    # =========================================================================
    # Model configs
    # =========================================================================

    group_kfold = GroupKFold(n_splits=5)

    model_configs = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=10000, random_state=args.seed),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=args.seed, n_jobs=1),
            "param_grid": {
                "clf__n_estimators": [300, 500, 1000],
                "clf__max_depth": [3, 6, 10],
                "clf__min_samples_leaf": [1, 5],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=args.seed,
                n_jobs=1,
                eval_metric="logloss",
                verbosity=0,
            ),
            "param_grid": {
                "clf__n_estimators": [300, 500, 1000],
                "clf__max_depth": [3, 6, 10],
                "clf__learning_rate": [0.001, 0.01, 0.1],
            },
        },
        "MLP": {
            "model": MLPClassifier(
                random_state=args.seed, max_iter=10000
            ),
            "param_grid": {
                "clf__hidden_layer_sizes": [(128,), (256,), (512,), (1024)],
                "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "clf__learning_rate_init": [1e-3, 1e-4],
                "clf__early_stopping": [True, False],
            },
        },
        "Transformer": {
            "model": _build_transformer_config(transformer_group_sizes),
            "param_grid": {
                "clf__module__d_model": [256, 512],
                "clf__module__n_heads": [4],
                "clf__module__n_transformer_layers": [1, 2, 3],
                "clf__module__dropout": [0.1, 0.3],
                "clf__optimizer__lr": [1e-3, 1e-4],
                "clf__max_epochs": [500],
            },
        },
    }

    # =========================================================================
    # Train models
    # =========================================================================

    print(f"\n{'=' * 70}")
    print("TRAINING MODELS (F1-optimized via GroupKFold CV)")
    print(f"{'=' * 70}")

    results = {}
    estimators = {}
    best_model_name = None
    best_val_f1 = 0.0
    best_estimator = None

    for name, config in model_configs.items():
        print(f"\n--- {name} ---")

        pipeline, param_grid = build_feature_pipeline(
            feature_names, scale_features, config["model"], config["param_grid"]
        )

        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=group_kfold.split(X_train, y_train, groups=qids_train),
            scoring="f1",
            n_jobs=8,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        y_proba_val = best.predict_proba(X_val)[:, 1]

        # Compute metrics at optimal threshold (on val)
        val_metrics = compute_metrics_at_threshold(y_val, y_proba_val, threshold=0.5)

        # Also find optimal threshold for F1 on val
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        opt_idx = np.argmax(f1_scores)
        opt_threshold = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5
        opt_metrics = compute_metrics_at_threshold(y_val, y_proba_val, threshold=opt_threshold)

        print(f"  Best params: {search.best_params_}")
        print(f"  Best CV F1: {search.best_score_:.4f}")
        print(f"  Val F1 @ 0.5: {val_metrics['f1']:.4f}")
        print(f"  Val F1 @ optimal (t={opt_threshold:.3f}): {opt_metrics['f1']:.4f}")
        print(f"  Val AUROC: {opt_metrics['auroc']:.4f}")
        print(f"  Val AUPRC: {opt_metrics['auprc']:.4f}")
        print(f"  Val TPR@5%FPR: {opt_metrics['tpr_at_5fpr']:.4f}")
        print(f"  Val Accuracy: {opt_metrics['accuracy']:.4f}")

        results[name] = {
            "best_params": search.best_params_,
            "cv_f1": float(search.best_score_),
            "val_metrics": opt_metrics,
            "optimal_threshold": float(opt_threshold),
        }
        estimators[name] = best

        # Track best model by F1 on validation
        if opt_metrics['f1'] > best_val_f1:
            best_val_f1 = opt_metrics['f1']
            best_model_name = name
            best_estimator = best

    # =========================================================================
    # Save best model
    # =========================================================================

    print(f"\n{'=' * 70}")
    print(f"BEST MODEL: {best_model_name} (Val F1: {best_val_f1:.4f})")
    print(f"{'=' * 70}")

    # Save model
    model_path = save_dir / "detector.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_estimator,
            "feature_names": feature_names,
            "scale_features": scale_features,
            "optimal_threshold": results[best_model_name]["optimal_threshold"],
            "model_name": best_model_name,
            "train_years": args.train_years,
        }, f)
    print(f"Saved best model to: {model_path}")

    # Save every candidate family's best estimator so 4.0/5.0 can score and
    # report each one individually. ``detector.pkl`` (above) stays as the
    # overall-best alias for pipeline pre-flight / skip checks.
    for name, estimator in estimators.items():
        family_path = save_dir / f"detector_{name}.pkl"
        with open(family_path, "wb") as f:
            pickle.dump({
                "model": estimator,
                "feature_names": feature_names,
                "scale_features": scale_features,
                "optimal_threshold": results[name]["optimal_threshold"],
                "model_name": name,
                "train_years": args.train_years,
            }, f)
        print(f"Saved {name} detector to: {family_path}")

    # Save validation results
    results_path = save_dir / "val_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved val results to: {results_path}")

    # Save summary
    summary = {
        "best_model": best_model_name,
        "best_val_f1": best_val_f1,
        "train_tokens": int(len(y_train)),
        "val_tokens": int(len(y_val)),
        "train_hallucination_rate": float(y_train.mean()),
        "val_hallucination_rate": float(y_val.mean()),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "detectors": [
            {
                "family": name,
                "val_f1": results[name]["val_metrics"]["f1"],
                "optimal_threshold": results[name]["optimal_threshold"],
                "is_best": name == best_model_name,
            }
            for name in model_configs
        ],
    }
    summary_path = save_dir / "train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    print("\nTraining complete! Use 4.0-model-evaluation.py for OOD evaluation.")
