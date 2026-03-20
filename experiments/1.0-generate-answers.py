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
    generate_params, 
    standardize_outputs, 
    move_to_device,
    tokenize_realtimeqa,
)
from moeuncert.monitoring import MoEMonitor
from moeuncert.metrics import compute_metrics
from moeuncert.experiments import resolve_model_slug, resolve_dataset_slug

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_batch_generation(tokenized_dataset, model, tokenizer, batch_size=2, save_path=None):
    """Run batched generation over a tokenized dataset, returning collated outputs."""
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
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            gen_params = generate_params(
                batch,
                tokenizer,
                max_new_tokens=65,
                output_attentions=True,
                output_hidden_states=True,
                output_scores=True,
                output_router_logits=False,
            )

            input_ids = move_to_device(batch["input_ids"], device="cpu")
            if "input_ids" not in all_outputs:
                all_outputs["input_ids"] = []
            all_outputs["input_ids"].append(input_ids)
            del batch

            outputs = model.generate(**gen_params, do_sample=False)
            outputs_processed = standardize_outputs(outputs, device="cpu")
            outputs_processed = {
                "sequences": outputs_processed["sequences"],
                **compute_metrics(outputs_processed)
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


def read_and_collate_outputs(file_list, tokenizer, get_keys=None):
    """
    Read saved batch outputs and collate into a single dictionary of tensors.
    """

    # Ensure we have the necessary keys to extract generated answers
    if "generated_answer" in get_keys:
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
            pad_value = (
                tokenizer.pad_token_id if key in ("sequences", "input_ids") else 0
            )
            padded = []
            for t in value:
                pad_cfg = []
                for d in range(
                    t.dim() - 1, 0, -1
                ):  # F.pad pads from last dim backwards
                    pad_cfg += [0, max_sizes[d] - t.shape[d]]
                padded.append(torch.nn.functional.pad(t, pad_cfg, value=pad_value))
            all_outputs[key] = torch.concat(padded, dim=0)

    if "sequences" in all_outputs and "input_ids" in all_outputs:
        all_outputs["generated_answer_ids"] = all_outputs["sequences"][
            :, all_outputs["input_ids"].shape[-1] :
        ]
        all_outputs["generated_answer"] = tokenizer.batch_decode(
            all_outputs["sequences"][:, all_outputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
    return all_outputs


def compute_scores(candidates, references, bertscore, rouge, bleu):
    rouge_scores = rouge.compute(
        predictions=candidates, references=references, use_aggregator=False
    )
    bert_scores = bertscore.compute(
        predictions=candidates, references=references, lang="en", verbose=True
    )
    bleu_scores = [
        (
            bleu.compute(predictions=[candidate], references=[reference])["bleu"]
            if candidate
            and (reference if isinstance(reference, str) else all(reference))
            else np.nan
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
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
    outputs = model_monitor.generate(**inputs, max_new_tokens=65)
    print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])

    ###############################################################################
    # Batch generation for all questions in the dataset

    dataset = Dataset.from_pandas(df)
    batch_size = 6

    tokenized = dataset.map(
        lambda examples: tokenize_realtimeqa(tokenizer, examples, with_evidence=False),
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format(type="torch")

    base_gen_dir = out_dir / model_slug / "base_generation"
    base_gen_dir.mkdir(parents=True, exist_ok=True)
    run_batch_generation(tokenized, model, tokenizer, batch_size, save_path=base_gen_dir)
    print(f"Model outputs saved to {base_gen_dir}")

    #########################################################################################
    # Batch generation with evidence (RAG simulation)

    tokenized_rag = dataset.map(
        lambda examples: tokenize_realtimeqa(tokenizer, examples, with_evidence=True),
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_rag.set_format(type="torch")

    evidence_gen_dir = out_dir / model_slug / "evidence_generation"
    evidence_gen_dir.mkdir(parents=True, exist_ok=True)
    run_batch_generation(tokenized_rag, model, tokenizer, batch_size, save_path=evidence_gen_dir)
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

