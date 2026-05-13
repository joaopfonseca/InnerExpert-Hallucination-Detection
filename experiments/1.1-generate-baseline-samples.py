"""
Generate multiple sampled responses per question for sampling-based baselines.

This script produces N stochastically sampled answers per question, needed by:
- Semantic Uncertainty (Kuhn et al., 2023)
- Semantic Energy (Ma et al., 2025)
- SelfCheckGPT (Manakul et al., 2023)

Each question generates `num_samples` responses with `do_sample=True` and a
configurable temperature. All sampled responses, their sequences, and the
full output logits are saved for downstream baseline evaluation.

This script follows the same structure and conventions as 1.0-generate-answers.py.
"""

import argparse
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets import fetch_realtimeqa
from moeuncert.utils import (
    standardize_outputs,
    move_to_device,
    tokenize_realtimeqa,
)
from moeuncert.monitoring import MoEMonitor
from moeuncert.metrics import compute_baseline_features
from moeuncert.experiments import (
    resolve_model_slug,
    resolve_dataset_slug,
    get_quantization_kwargs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_batch_sampling(tokenized_dataset, model_monitor, tokenizer, batch_size=2,
                       save_path=None, max_new_tokens=65, num_samples=5,
                       temperature=0.7, top_p=0.9):
    """Run batched sampling generation, producing N responses per question.

    Parameters
    ----------
    tokenized_dataset : Dataset
        Tokenized dataset with input_ids and attention_mask.
    model_monitor : MoEMonitor
        Model wrapper with generate() method.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    batch_size : int
        Number of samples per batch.
    save_path : Path, optional
        Directory to save batch outputs.
    max_new_tokens : int
        Maximum number of new tokens to generate per sample.
    num_samples : int
        Number of stochastically sampled responses per question.
    temperature : float
        Sampling temperature for generation.
    top_p : float
        Nucleus sampling parameter.

    Returns
    -------
    dict
        Collated outputs with keys for each sample index.
    """
    all_outputs = {}
    total_batches = len(tokenized_dataset) // batch_size

    with torch.no_grad():
        for i in tqdm(list(range(0, len(tokenized_dataset), batch_size)),
                      desc="Sampling batches"):

            batch_idx = i // batch_size

            # Check if batch outputs already exist (resumption support)
            if save_path is not None:
                filename = (
                    f"sampled_outputs__batch_{batch_idx}"
                    f"_of_{total_batches}"
                    f"__batch_size_{batch_size}"
                    f"__num_samples_{num_samples}.pt"
                )
                if (save_path / filename).exists():
                    print(f"Batch {batch_idx} already exists, skipping.")
                    continue

            batch = tokenized_dataset[i:i + batch_size]
            batch_size_actual = len(batch["input_ids"])

            # Extract question_id
            question_ids = batch["question_id"]
            del batch["question_id"]

            # Store question IDs once (shared across all samples)
            if "question_id" not in all_outputs:
                all_outputs["question_id"] = []
            all_outputs["question_id"].extend(question_ids)

            # For each sample, generate stochastically
            for sample_idx in range(num_samples):
                batch_on_device = move_to_device(batch, device=DEVICE)

                outputs = model_monitor.generate(
                    **batch_on_device,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                del batch_on_device

                outputs_processed = standardize_outputs(outputs, device=DEVICE)
                # Compute only logit-level features (skips SVD hidden/attention scores)
                features = compute_baseline_features(outputs_processed)
                outputs_processed.update(features)
                # Keep only what downstream baselines need, move to CPU
                keep_keys = {"sequences", "scores", "log_likelihoods",
                             "entropies", "perplexity"}
                outputs_processed = move_to_device(
                    {k: v for k, v in outputs_processed.items() if k in keep_keys},
                    device="cpu",
                )
                del outputs

                # Store each key with a sample index suffix
                for key, output in outputs_processed.items():
                    sample_key = f"{key}_sample{sample_idx}"
                    if sample_key not in all_outputs:
                        all_outputs[sample_key] = []
                    all_outputs[sample_key].append(output)
                    del output
                del outputs_processed

            # Save to disk and clear memory
            if save_path is not None:
                filename = (
                    f"sampled_outputs__batch_{i // batch_size}"
                    f"_of_{total_batches}"
                    f"__batch_size_{batch_size}"
                    f"__num_samples_{num_samples}.pt"
                )
                torch.save(all_outputs, save_path / filename)
                del all_outputs
                all_outputs = {}

    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multiple sampled responses per question for "
                    "sampling-based hallucination detection baselines."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name (default: allenai/OLMoE-1B-7B-0924-Instruct)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="4-bit",
        choices=["16-bit", "8-bit", "4-bit"],
        help="Quantization level for the model.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Year(s) for RealtimeQA, e.g. --years 2025 2026. "
             "Defaults to previous month year.",
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
        "--max-new-tokens",
        type=int,
        default=65,
        help="Maximum number of new tokens to generate per answer (default: 65).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of stochastically sampled responses per question (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for generation (default: 2).",
    )
    args = parser.parse_args()

    print(f"Device set to: {DEVICE}")
    print(f"Generating {args.num_samples} samples per question "
          f"(temperature={args.temperature}, top_p={args.top_p})")

    model_name = args.model
    model_slug = resolve_model_slug(model_name)
    years, month, dataset_slug = resolve_dataset_slug(args.years, args.month)

    # Build output directory and load (or fetch) the dataset
    out_dir = Path("data") / dataset_slug
    if not out_dir.exists():
        if len(years) == 1 and month is not None:
            df = fetch_realtimeqa(split=years[0], month=month)
        else:
            df = pd.concat([fetch_realtimeqa(split=y) for y in years])
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "realtimeqa_original.parquet", index=False)
    else:
        df = pd.read_parquet(out_dir / "realtimeqa_original.parquet")

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_kwargs = get_quantization_kwargs(args.quantize)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        device_map="auto",
        **quantization_kwargs,
    )
    model_monitor = MoEMonitor(
        model=model,
        tokenizer=tokenizer,
        output_router_logits=False,
        output_experts_hidden=False,
    )

    # Tokenize dataset
    dataset = Dataset.from_pandas(df)
    tokenized = dataset.map(
        lambda examples: tokenize_realtimeqa(
            tokenizer, examples, with_evidence=False
        ),
        batched=True,
        remove_columns=[col for col in dataset.column_names
                        if col != "question_id"],
    )
    tokenized.set_format(type="torch")

    # Output directory for sampled responses
    sample_dir = out_dir / model_slug / "sampled_generation"
    sample_dir.mkdir(parents=True, exist_ok=True)

    run_batch_sampling(
        tokenized,
        model_monitor,
        tokenizer,
        batch_size=args.batch_size,
        save_path=sample_dir,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"Sampled outputs saved to {sample_dir}")
    print(f"Keys per sample: sequences, scores, log_likelihoods, entropies, "
          f"hidden_scores, attention_scores, etc.")
