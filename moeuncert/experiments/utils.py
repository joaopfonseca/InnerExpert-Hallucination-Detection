import numpy as np
import torch
from sklearn.metrics import roc_curve


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
