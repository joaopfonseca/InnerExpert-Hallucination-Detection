import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Set, Union
from sklearn.metrics import roc_curve


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


def optimal_threshold(y_true, scores):
    """Return the score threshold that maximises accuracy."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    acc = (tpr * n_pos + (1 - fpr) * n_neg) / len(y_true)
    best = np.argmax(acc)
    return thresholds[best], acc[best]
