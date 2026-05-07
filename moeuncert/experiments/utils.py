import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Set, Union, Tuple
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score, accuracy_score

logger = logging.getLogger(__name__)


def find_generation_boundaries(
    input_ids: torch.Tensor,
    sequences: torch.Tensor,
) -> Tuple[int, int]:
    """Find where the generated answer starts and ends in the full sequence.

    Handles left-padded sequences by detecting the prompt content region
    from input_ids, then computing the generated region in sequences.

    Parameters
    ----------
    input_ids : torch.Tensor
        Token IDs for the prompt (1D, with left padding).
    sequences : torch.Tensor
        Full generated sequence (1D, prompt + generation, with padding).

    Returns
    -------
    Tuple[int, int]
        (gen_start, gen_end) positions in sequences.
    """
    pad_token = input_ids[0].item()
    non_pad_mask = input_ids != pad_token
    input_content_start = (
        torch.where(non_pad_mask)[0][0].item()
        if torch.any(non_pad_mask) else len(input_ids)
    )

    non_zero_mask = input_ids != 0
    input_content_end = (
        torch.where(non_zero_mask)[0][-1].item() + 1
        if torch.any(non_zero_mask) else 0
    )

    seq_non_pad_mask = sequences != pad_token
    seq_content_start = (
        torch.where(seq_non_pad_mask)[0][0].item()
        if torch.any(seq_non_pad_mask) else len(sequences)
    )

    content_len = input_content_end - input_content_start
    gen_start = seq_content_start + content_len

    seq_pad_tokens = torch.where(sequences[gen_start:] == pad_token)[0]
    if len(seq_pad_tokens) > 0:
        gen_end = gen_start + seq_pad_tokens[0].item()
    else:
        logger.warning(
            "find_generation_boundaries: no pad token ID %d found after gen_start=%d; "
            "falling back to len(sequences)=%d. Generation boundary may be too long. "
            "This could indicate unexpected tokenizer behavior.",
            pad_token, gen_start, len(sequences),
        )
        gen_end = len(sequences)

    return gen_start, gen_end


def read_and_collate_outputs(
    file_list: Union[List[Path], List[str]], 
    tokenizer=None, 
    get_keys: Optional[Set[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Read saved batch outputs and collate into a single dictionary of tensors.
    
    This function loads .pt files containing batch outputs from model generation
    and concatenates them along the batch dimension, handling variable-length
    sequences with padding when necessary.
    
    Parameters
    ----------
    file_list : List[Path] or List[str]
        List of .pt files containing batch outputs
    tokenizer : transformers.PreTrainedTokenizer, optional
        Tokenizer for padding sequences. If provided, will use tokenizer.pad_token_id
        for padding 'sequences' and 'input_ids'. Also enables decoding of generated text.
    get_keys : Set[str], optional
        If provided, only load these keys from batch files. Note: if 'generated_answer'
        is requested, 'sequences' and 'input_ids' are automatically included.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Collated outputs with all batches concatenated along dimension 0.
        If tokenizer is provided and both 'sequences' and 'input_ids' are present,
        also includes 'generated_answer_ids' and 'generated_answer' (decoded text).
    
    Examples
    --------
    >>> from pathlib import Path
    >>> from transformers import AutoTokenizer
    >>> 
    >>> # Load all outputs
    >>> files = list(Path("outputs/").glob("batch_*.pt"))
    >>> outputs = read_and_collate_outputs(files)
    >>> 
    >>> # Load only specific keys
    >>> outputs = read_and_collate_outputs(files, get_keys={'hidden_scores', 'sequences'})
    >>> 
    >>> # With tokenizer for text decoding
    >>> tokenizer = AutoTokenizer.from_pretrained("model-name")
    >>> outputs = read_and_collate_outputs(files, tokenizer=tokenizer)
    """
    # Ensure we have the necessary keys to extract generated answers
    if get_keys is not None and "generated_answer" in get_keys:
        get_keys = set(get_keys) | {"sequences", "input_ids"}

    all_outputs = {}
    for filepath in file_list:
        batch_outputs = torch.load(filepath, map_location="cpu")
        for key, value in batch_outputs.items():
            if get_keys is not None and key not in get_keys:
                continue
            if key not in all_outputs:
                all_outputs[key] = []
            all_outputs[key].extend(value)

    for key, value in all_outputs.items():
        # Leave non-tensor values (e.g. question_id list of strings) as-is
        if not isinstance(value[0], torch.Tensor):
            all_outputs[key] = value
            continue
        if all(v.shape == value[0].shape for v in value):
            all_outputs[key] = torch.concat(value, dim=0)
        else:
            # Pad to max size in each dimension before concatenating (e.g. variable
            # generation lengths across batches when early stopping occurs)
            max_sizes = [max(v.shape[d] for v in value) for d in range(value[0].dim())]
            pad_value = 0
            if tokenizer is not None and key in ("sequences", "input_ids"):
                pad_value = tokenizer.pad_token_id
            padded = []
            for t in value:
                pad_cfg = []
                for d in range(
                    t.dim() - 1, 0, -1
                ):  # F.pad pads from last dim backwards
                    pad_cfg += [0, max_sizes[d] - t.shape[d]]
                padded.append(torch.nn.functional.pad(t, pad_cfg, value=pad_value))
            all_outputs[key] = torch.concat(padded, dim=0)

    if tokenizer is not None and "sequences" in all_outputs and "input_ids" in all_outputs:
        all_outputs["generated_answer_ids"] = all_outputs["sequences"][
            :, all_outputs["input_ids"].shape[-1] :
        ]
        all_outputs["generated_answer"] = tokenizer.batch_decode(
            all_outputs["sequences"][:, all_outputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
    return all_outputs


def get_quantization_kwargs(quantize: str) -> dict:
    """Get model loading kwargs for the specified quantization level.

    Args:
        quantize: Quantization level string, one of "16-bit", "8-bit", or "4-bit".

    Returns:
        Dictionary of kwargs to pass to AutoModelForCausalLM.from_pretrained().
    """
    if quantize == "16-bit":
        return {"torch_dtype": torch.float16}
    elif quantize == "8-bit":
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "torch_dtype": torch.float16,
        }
    elif quantize == "4-bit":
        from transformers import BitsAndBytesConfig
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            "torch_dtype": torch.float16,
        }
    else:
        raise ValueError(f"Unknown quantization level: {quantize}. Choose from '16-bit', '8-bit', or '4-bit'.")


def optimal_threshold(y_true, scores, metric="f1"):
    """Return the score threshold that maximises the specified metric.

    Args:
        y_true: True binary labels.
        scores: Predicted scores.
        metric: Metric to optimize. "f1" or "accuracy". Default "f1".

    Returns:
        Tuple of (optimal_threshold, best_metric_value).
    """
    from sklearn.metrics import precision_recall_curve

    if metric == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        # precision_recall_curve returns len(thresholds) + 1 precision/recall points,
        # so only the first len(thresholds) F1 values correspond to actual thresholds.
        threshold_f1_scores = f1_scores[:-1]
        if len(thresholds) == 0:
            return 0.5, f1_scores[0]
        best_idx = np.argmax(threshold_f1_scores)
        best_threshold = thresholds[best_idx]
        best_value = threshold_f1_scores[best_idx]
        return best_threshold, best_value
    elif metric == "accuracy":
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
        acc = (tpr * n_pos + (1 - fpr) * n_neg) / len(y_true)
        best_idx = np.argmax(acc)
        return thresholds[best_idx], acc[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'f1' or 'accuracy'.")


def stratified_group_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split grouped by question_id.

    Each question (and all its tokens/responses) goes entirely to train or val.
    Stratification ensures similar label distribution in train and val.

    Parameters
    ----------
    y : np.ndarray
        Labels (binary).
    groups : np.ndarray
        Group IDs (question_id values).
    test_size : float
        Fraction of groups for validation.
    random_state : int
        Random seed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Boolean masks for train and val indices.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    unique_groups = np.unique(groups)
    group_labels = np.array([
        int(y[groups == g].max()) for g in unique_groups
    ])

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_group_idx, val_group_idx = next(
        sss.split(unique_groups, group_labels)
    )

    train_groups = unique_groups[train_group_idx]
    val_groups = unique_groups[val_group_idx]

    train_mask = np.isin(groups, train_groups)
    val_mask = np.isin(groups, val_groups)

    return train_mask, val_mask


def create_token_labels(
    generated_text: str,
    hallucinated_spans,
    offset_mapping: List[Tuple[int, int]],
) -> torch.Tensor:
    """Create binary token labels based on hallucinated text spans.

    Uses the tokenizer's offset mapping to accurately map character-level
    hallucinated spans to token-level binary labels. A token is labeled 1
    (hallucinated) if any part of its character span overlaps with any
    hallucinated span.

    Parameters
    ----------
    generated_text : str
        The generated answer text.
    hallucinated_spans : array-like
        List of hallucinated text substrings, or None/empty.
    offset_mapping : List[Tuple[int, int]]
        Tokenizer offset mapping for the generated_text.

    Returns
    -------
    torch.Tensor
        Binary token labels (1 = hallucinated, 0 = grounded).
    """
    n_tokens = len(offset_mapping)
    token_labels = torch.zeros(n_tokens, dtype=torch.long)

    if hallucinated_spans is None:
        return token_labels
    if isinstance(hallucinated_spans, (float, np.floating)) and np.isnan(hallucinated_spans):
        return token_labels
    if isinstance(hallucinated_spans, np.ndarray) and len(hallucinated_spans) == 0:
        return token_labels
    if isinstance(hallucinated_spans, (list, tuple, set)) and len(hallucinated_spans) == 0:
        return token_labels
    if not isinstance(hallucinated_spans, (np.ndarray, list, tuple, set)):
        return token_labels

    for span_text in hallucinated_spans:
        span_text = str(span_text)
        span_start = generated_text.find(span_text)
        if span_start == -1:
            continue
        span_end = span_start + len(span_text)

        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_end > span_start and char_start < span_end:
                token_labels[token_idx] = 1

    return token_labels


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics at a given threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities.
    threshold : float
        Classification threshold.

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics (auroc, auprc, f1, accuracy, tpr_at_5fpr).
    """
    y_pred = (y_proba >= threshold).astype(int)

    # AUROC (threshold-independent)
    auroc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5

    # AUPRC
    auprc = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0

    # F1
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # TPR @ 5% FPR
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        tpr_at_5fpr = np.interp(0.05, fpr, tpr)
    else:
        tpr_at_5fpr = 0.0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "accuracy": float(acc),
        "tpr_at_5fpr": float(tpr_at_5fpr),
        "threshold": float(threshold),
    }
