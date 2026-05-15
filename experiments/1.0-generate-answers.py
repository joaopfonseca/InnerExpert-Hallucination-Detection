"""
This script demonstrates how to use the OLMoE-1B-7B model for generating
answers to questions in the RealtimeQA dataset, both with and without evidence
(RAG simulation).

It can be used to save the generated outputs for later analysis.
"""

import argparse
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import evaluate

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
from moeuncert.metrics import compute_metrics
from moeuncert.experiments import (
    resolve_model_slug, 
    resolve_dataset_slug, 
    get_quantization_kwargs,
    read_and_collate_outputs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_batch_generation(tokenized_dataset, model_monitor, tokenizer, batch_size=2, save_path=None, max_new_tokens=65,
                         return_baseline_features=False):
    """Run batched generation over a tokenized dataset, returning collated outputs.
    
    Parameters
    ----------
    tokenized_dataset : Dataset
        Tokenized dataset with input_ids and attention_mask
    model_monitor : ModelMonitor
        Model wrapper with generate() method
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model
    batch_size : int
        Number of samples per batch
    save_path : Path, optional
        Directory to save batch outputs
    max_new_tokens : int
        Maximum number of new tokens to generate per sample
    """
    all_outputs = {}
    with torch.no_grad():
        for i in tqdm(list(range(0, len(tokenized_dataset), batch_size))):

            # Check if batch outputs already exist (in case of re-running after an interruption)
            if save_path is not None:
                filename = (
                    f"model_outputs__batch_{i//batch_size}"
                    f"_of_{len(tokenized_dataset)//batch_size}"
                    f"__batch_size_{batch_size}.pt"
                )
                if (save_path / filename).exists():
                    print(f"Batch {i//batch_size} already exists.")
                    continue

            batch = tokenized_dataset[i : i + batch_size]
            
            # Extract question_id
            question_ids = batch["question_id"]
            if "question_id" not in all_outputs:
                all_outputs["question_id"] = []
            all_outputs["question_id"].append(question_ids)
            del batch["question_id"]

            batch["input_ids"] = tokenizer.pad(
                {"input_ids": batch["input_ids"]}, padding=True, return_tensors="pt"
            )["input_ids"]
            batch["attention_mask"] = tokenizer.pad(
                {"input_ids": batch["attention_mask"]}, padding=True, return_tensors="pt"
            )["input_ids"]

            input_ids = move_to_device(batch["input_ids"], device="cpu")
            if "input_ids" not in all_outputs:
                all_outputs["input_ids"] = []
            all_outputs["input_ids"].append(input_ids)
            
            batch = move_to_device(batch, device=DEVICE)
            outputs = model_monitor.generate(**batch, max_new_tokens=max_new_tokens)
            del batch

            outputs_processed = standardize_outputs(outputs, device="cpu")
            outputs_processed = {
                "sequences": outputs_processed["sequences"],
                **compute_metrics(outputs_processed, return_baseline_features=return_baseline_features)
            }

            del outputs
            for key, output in outputs_processed.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(output)
                del output
            del outputs_processed

            # This is done to save memory if the dataset is large
            if save_path is not None:
                torch.save(all_outputs, save_path / filename)
                # Clear from memory after saving
                del all_outputs
                all_outputs = {}

    return all_outputs


def compute_scores(candidates, references, bertscore, rouge, bleu):
    safe_candidates = [c if c and c.strip() else "." for c in candidates]
    safe_references = [
        [r for r in ref if r and r.strip()] or ["."]
        for ref in references
    ]

    rouge_scores = rouge.compute(
        predictions=candidates, references=references, use_aggregator=False
    )
    bert_scores = bertscore.compute(
        predictions=safe_candidates, references=safe_references, lang="en", verbose=True
    )
    bleu_scores = [
        (
            bleu.compute(predictions=[candidate], references=[reference])["bleu"]
            if candidate
            and (reference if isinstance(reference, str) else all(reference))
            else 0.0
        )
        for candidate, reference in zip(candidates, references)
    ]
    return {
        **rouge_scores,
        **{
            f"bert_{key}": value
            for key, value in bert_scores.items()
            if key != "hashcode"
        },
        "bleu": bleu_scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate answers for RealtimeQA using an instruction-tuned LLM."
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
        help="Quantization level for the model."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Year(s) for RealtimeQA, e.g. --years 2025 2026. Defaults to previous month year.",
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
        "--return-baseline-features",
        action="store_true",
        help="If set, also compute log-likelihoods and full-vocabulary entropies "
             "needed by trainable baselines (e.g., HaluNet).",
    )
    args = parser.parse_args()

    print(f"Device set to: {DEVICE}")

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

    # Clear cache to ensure we have enough memory for the model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Required for batching
    tokenizer.padding_side = "left"  # Left padding for generation
    quantization_kwargs = get_quantization_kwargs(args.quantize)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        device_map="auto",
        **quantization_kwargs,
    )
    model_monitor = MoEMonitor(model=model, tokenizer=tokenizer, output_router_logits=False)

    ###############################################################################
    # Single question generation example

    rand_idx = np.random.choice(len(df))
    inputs = tokenize_realtimeqa(
        tokenizer, df.iloc[rand_idx : rand_idx + 1],
        with_evidence=False
    )
    inputs = move_to_device(inputs, device=DEVICE)

    # NOTE: output_router_logits must be False for `generate`; This is a known limitation.
    # See: https://github.com/huggingface/transformers/issues/30731
    outputs = model_monitor.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])

    ###############################################################################
    # Batch generation for all questions in the dataset

    dataset = Dataset.from_pandas(df)
    batch_size = 6

    tokenized = dataset.map(
        lambda examples: tokenize_realtimeqa(tokenizer, examples, with_evidence=False),
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != "question_id"],
    )
    tokenized.set_format(type="torch")

    base_gen_dir = out_dir / model_slug / "base_generation"
    base_gen_dir.mkdir(parents=True, exist_ok=True)
    run_batch_generation(
        tokenized, 
        model_monitor, 
        tokenizer, 
        batch_size, 
        save_path=base_gen_dir,
        max_new_tokens=args.max_new_tokens,
        return_baseline_features=args.return_baseline_features,
    )
    print(f"Model outputs saved to {base_gen_dir}")

    #########################################################################################
    # Batch generation with evidence (RAG simulation)

    tokenized_rag = dataset.map(
        lambda examples: tokenize_realtimeqa(tokenizer, examples, with_evidence=True),
        batched=True,
        remove_columns=[col for col in dataset.column_names if col != "question_id"],
    )
    tokenized_rag.set_format(type="torch")

    evidence_gen_dir = out_dir / model_slug / "evidence_generation"
    evidence_gen_dir.mkdir(parents=True, exist_ok=True)
    run_batch_generation(
        tokenized_rag, 
        model_monitor, 
        tokenizer, 
        batch_size, 
        save_path=evidence_gen_dir,
        max_new_tokens=args.max_new_tokens,
        return_baseline_features=args.return_baseline_features,
    )
    print(f"Evidence-based outputs saved to {evidence_gen_dir}")

    #########################################################################################
    # Compute metrics
    # pip install evaluate rouge_score

    all_outputs = read_and_collate_outputs(
        sorted(
            base_gen_dir.iterdir(),
            key=lambda p: int(p.stem.split("batch_")[1].split("_of_")[0]),
        ),
        tokenizer,
        get_keys=["generated_answer"],
    )
    df["generated_answer"] = all_outputs["generated_answer"]

    all_outputs_rag = read_and_collate_outputs(
        sorted(
            evidence_gen_dir.iterdir(),
            key=lambda p: int(p.stem.split("batch_")[1].split("_of_")[0]),
        ),
        tokenizer,
        get_keys=["generated_answer"],
    )
    df["generated_answer_rag"] = all_outputs_rag["generated_answer"]

    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("evaluate-metric/bleu")

    # Sometimes there is no evidence
    references = [
        [ev for ev in (gt, evidence) if len(ev.strip()) > 0]
        for gt, evidence in zip(df["answer_str"], df["evidence"])
    ]
    candidates = all_outputs["generated_answer"]
    candidates_rag = all_outputs_rag["generated_answer"]

    scores = compute_scores(candidates, references, bertscore, rouge, bleu)

    # Add per-sample scores to the dataframe
    score_cols = {k: v for k, v in scores.items() if isinstance(v, list)}
    for col, values in score_cols.items():
        df[col] = values

    # Compute and store metrics for RAG version
    scores_rag = compute_scores(candidates_rag, references, bertscore, rouge, bleu)

    score_cols_rag = {k: v for k, v in scores_rag.items() if isinstance(v, list)}
    for col, values in score_cols_rag.items():
        df[f"{col}_rag"] = values

    # Save the dataframe (questions, answers, and per-sample scores) as Parquet
    df.to_parquet(out_dir / model_slug / "results.parquet", index=False)
    print(f"DataFrame saved to {out_dir / model_slug / 'results.parquet'}")

