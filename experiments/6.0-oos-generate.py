"""6.0 — OOS answer generation + sampled responses.

Generates LLM answers for an out-of-sample dataset (SQuAD, TruthfulQA,
NQ-Open, FreshQA) using the same MoE-instrumentation pipeline as
``1.0-generate-answers.py``, but with dataset-specific prompt
construction via ``moeuncert.datasets_adapters``.

For datasets with an evidence field (SQuAD, FreshQA), runs two
passes: base (no evidence) and evidence (with evidence).  For datasets
without evidence (TruthfulQA, NQ-Open), runs base-only.

Optionally generates sampled responses (``--num-samples > 0``) for
Semantic Uncertainty / Semantic Energy / SelfCheckGPT.

Outputs:
  - data/<dataset_slug>/<model_slug>/base_generation/model_outputs__batch_*.pt
  - data/<dataset_slug>/<model_slug>/evidence_generation/model_outputs__batch_*.pt
  - data/<dataset_slug>/<model_slug>/sampled_generation/sampled_outputs__batch_*.pt
  - data/<dataset_slug>/<model_slug>/results.parquet

Usage:
    python 6.0-oos-generate.py --dataset squad --model allenai/OLMoE-1B-7B-0924-Instruct
    python 6.0-oos-generate.py --dataset truthfulqa --num-samples 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm

try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets_adapters import get_adapter, list_oos_datasets
from moeuncert.utils import standardize_outputs, move_to_device
from moeuncert.monitoring import MoEMonitor
from moeuncert.metrics import compute_metrics
from moeuncert.metrics._metrics import compute_baseline_features
from moeuncert.experiments import (
    resolve_model_slug,
    resolve_cache_dir,
    get_quantization_kwargs,
    read_and_collate_outputs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Generation (reuses 1.0's run_batch_generation logic)
# ---------------------------------------------------------------------------


def run_batch_generation(
    tokenized_dataset,
    model_monitor,
    tokenizer,
    batch_size=2,
    save_path=None,
    max_new_tokens=65,
    return_baseline_features=False,
):
    """Batched generation over a tokenized dataset. Mirrors 1.0's logic."""
    all_outputs = {}
    with torch.no_grad():
        for i in tqdm(list(range(0, len(tokenized_dataset), batch_size))):
            if save_path is not None:
                filename = (
                    f"model_outputs__batch_{i // batch_size}"
                    f"_of_{len(tokenized_dataset) // batch_size}"
                    f"__batch_size_{batch_size}.pt"
                )
                if (save_path / filename).exists():
                    print(f"Batch {i // batch_size} already exists.")
                    continue

            batch = tokenized_dataset[i : i + batch_size]
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
                **compute_metrics(
                    outputs_processed, return_baseline_features=return_baseline_features
                ),
            }

            del outputs
            for key, output in outputs_processed.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(output)
                del output
            del outputs_processed

            if save_path is not None:
                torch.save(all_outputs, save_path / filename)
                del all_outputs
                all_outputs = {}

    return all_outputs


# ---------------------------------------------------------------------------
# Sampling (reuses 1.1's run_batch_sampling logic)
# ---------------------------------------------------------------------------


def run_batch_sampling(
    tokenized_dataset,
    model_monitor,
    tokenizer,
    batch_size=2,
    save_path=None,
    max_new_tokens=65,
    num_samples=5,
    temperature=0.7,
    top_p=0.9,
):
    """Batched sampling generation, producing N responses per question."""
    all_outputs = {}
    total_batches = len(tokenized_dataset) // batch_size

    with torch.no_grad():
        for i in tqdm(
            list(range(0, len(tokenized_dataset), batch_size)),
            desc="Sampling batches",
        ):
            batch_idx = i // batch_size
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

            batch = tokenized_dataset[i : i + batch_size]
            question_ids = batch["question_id"]
            del batch["question_id"]

            if "question_id" not in all_outputs:
                all_outputs["question_id"] = []
            all_outputs["question_id"].extend(question_ids)

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
                del outputs

                outputs_processed["stop_token_id"] = tokenizer.eos_token_id
                features = compute_baseline_features(outputs_processed)
                valid_mask = features.pop("valid_token_mask")
                gen_len = features["log_likelihoods"].shape[1]
                generated_ids = outputs_processed["sequences"][:, -gen_len:]
                scores = outputs_processed["scores"]
                generated_ids = generated_ids.masked_fill(~valid_mask, 0)
                scores = scores.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

                keep = {
                    "sequences": generated_ids,
                    "scores": scores,
                    "log_likelihoods": features["log_likelihoods"],
                    "entropies": features["entropies"],
                    "perplexity": features["perplexity"],
                }
                outputs_processed = move_to_device(keep, device="cpu")
                del keep, features, generated_ids, scores, valid_mask

                for key, output in outputs_processed.items():
                    sample_key = f"{key}_sample{sample_idx}"
                    if sample_key not in all_outputs:
                        all_outputs[sample_key] = []
                    all_outputs[sample_key].append(output)
                    del output
                del outputs_processed

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


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------


def tokenize_with_adapter(tokenizer, df, adapter, with_evidence=False):
    """Tokenize the common-schema DataFrame via the adapter's prompt builder.

    Returns a datasets.Dataset-like dict with input_ids, attention_mask,
    and question_id columns.
    """
    dataset = Dataset.from_pandas(df)
    tokenized = dataset.map(
        lambda examples: adapter.tokenize(tokenizer, pd.DataFrame(dict(examples)), with_evidence=with_evidence),
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "question_id"],
    )
    tokenized.set_format(type="torch")
    return tokenized


# ---------------------------------------------------------------------------
# Score computation (reuses 1.0's compute_scores logic)
# ---------------------------------------------------------------------------


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
        predictions=safe_candidates, references=safe_references, lang="en", verbose=False
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate OOS dataset answers with MoE instrumentation (6.0)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list_oos_datasets(),
        help="OOS dataset name",
    )
    parser.add_argument(
        "--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--quantize", type=str, default="4-bit",
        choices=["16-bit", "8-bit", "4-bit"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=65,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=6,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--num-samples", type=int, default=0,
        help="Number of sampled responses per question (0 = skip sampling)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (used when --num-samples > 0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling top_p (used when --num-samples > 0)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--return-baseline-features", action="store_true", default=True,
        help="Compute log_likelihoods/entropies/perplexity/last_hidden_states (default: on)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Randomly sample at most N questions from the dataset (default: all)",
    )

    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"6.0 — OOS GENERATION ({args.dataset})")
    print(f"{'=' * 70}")

    adapter = get_adapter(args.dataset)
    print(f"Dataset: {adapter.name} (slug={adapter.slug})")
    print(f"Has evidence: {adapter.has_evidence}")
    print(f"Task type: {adapter.task_type}")
    print(f"Model: {args.model}")

    # --- Fetch dataset ----------------------------------------------------
    print("\n[1/5] Fetching dataset...")
    df_raw = adapter.fetch()
    df = adapter.to_common_schema(df_raw)
    print(f"  {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    if args.max_samples is not None and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(df)} rows (max_samples={args.max_samples})")

    # --- Setup model ------------------------------------------------------
    print("\n[2/5] Loading model + tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=str(resolve_cache_dir(args.model))
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch.cuda.empty_cache()
    quantization_kwargs = get_quantization_kwargs(args.quantize)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",
        device_map="auto",
        cache_dir=str(resolve_cache_dir(args.model)),
        **quantization_kwargs,
    )
    model_monitor = MoEMonitor(model=model, tokenizer=tokenizer, output_router_logits=False)

    # --- Output directories -----------------------------------------------
    model_slug = resolve_model_slug(args.model)
    out_dir = args.data_root / adapter.slug
    model_dir = out_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {model_dir}")

    # --- Generation passes ------------------------------------------------
    # XSum: evidence-only (summarization framing — document IS the input).
    # Datasets with has_evidence=True: base + evidence passes.
    # Datasets with has_evidence=False: base-only.
    run_base = True
    run_evidence = adapter.has_evidence

    if adapter.task_type == "summarization":
        # Summarization: no base pass (can't summarize without the document)
        run_base = False
        run_evidence = True

    if run_base:
        print("\n[3/5] Base generation (no evidence)...")
        tokenized = tokenize_with_adapter(tokenizer, df, adapter, with_evidence=False)
        base_dir = model_dir / "base_generation"
        base_dir.mkdir(parents=True, exist_ok=True)
        run_batch_generation(
            tokenized, model_monitor, tokenizer,
            batch_size=args.batch_size, save_path=base_dir,
            max_new_tokens=args.max_new_tokens,
            return_baseline_features=args.return_baseline_features,
        )
        print(f"  Base outputs saved to {base_dir}")
    else:
        print("\n[3/5] Skipping base generation (no base pass for this dataset)")

    if run_evidence:
        print("\n[4/5] Evidence generation (with evidence)...")
        tokenized_ev = tokenize_with_adapter(tokenizer, df, adapter, with_evidence=True)
        ev_dir = model_dir / "evidence_generation"
        ev_dir.mkdir(parents=True, exist_ok=True)
        run_batch_generation(
            tokenized_ev, model_monitor, tokenizer,
            batch_size=args.batch_size, save_path=ev_dir,
            max_new_tokens=args.max_new_tokens,
            return_baseline_features=args.return_baseline_features,
        )
        print(f"  Evidence outputs saved to {ev_dir}")
    else:
        print("\n[4/5] Skipping evidence generation (no evidence field)")

    # --- Sampled generation (optional) -----------------------------------
    if args.num_samples > 0:
        print(f"\n[4b/5] Sampled generation ({args.num_samples} samples)...")
        # Sample from the base pass (no evidence) — matches 1.1 behaviour
        if run_base:
            tokenized_s = tokenize_with_adapter(tokenizer, df, adapter, with_evidence=False)
        else:
            tokenized_s = tokenize_with_adapter(tokenizer, df, adapter, with_evidence=True)
        sampled_dir = model_dir / "sampled_generation"
        sampled_dir.mkdir(parents=True, exist_ok=True)
        run_batch_sampling(
            tokenized_s, model_monitor, tokenizer,
            batch_size=max(1, args.batch_size // 2),
            save_path=sampled_dir,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"  Sampled outputs saved to {sampled_dir}")
    else:
        print("\n[4b/5] Skipping sampled generation (num-samples=0)")

    # --- Compute ROUGE/BERTScore/BLEU + save results.parquet -------------
    print("\n[5/5] Computing scores + saving results.parquet...")
    import evaluate

    # Decode generated answers from saved batches
    df["generated_answer"] = ""
    df["generated_answer_rag"] = ""

    if run_base:
        all_outputs = read_and_collate_outputs(
            sorted(base_dir.iterdir(),
                   key=lambda p: int(p.stem.split("batch_")[1].split("_of_")[0])),
            tokenizer,
            get_keys=["generated_answer"],
        )
        df["generated_answer"] = all_outputs["generated_answer"]

    if run_evidence:
        all_outputs_ev = read_and_collate_outputs(
            sorted(ev_dir.iterdir(),
                   key=lambda p: int(p.stem.split("batch_")[1].split("_of_")[0])),
            tokenizer,
            get_keys=["generated_answer"],
        )
        df["generated_answer_rag"] = all_outputs_ev["generated_answer"]

    # Compute ROUGE/BERTScore/BLEU against ground-truth answer_str
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("evaluate-metric/bleu")

    references = [[str(ans)] for ans in df["answer_str"]]

    if run_base:
        candidates = df["generated_answer"]
        scores = compute_scores(candidates, references, bertscore, rouge, bleu)
        score_cols = {k: v for k, v in scores.items() if isinstance(v, list)}
        for col, values in score_cols.items():
            df[col] = values

    if run_evidence:
        candidates_rag = df["generated_answer_rag"]
        scores_rag = compute_scores(candidates_rag, references, bertscore, rouge, bleu)
        score_cols_rag = {k: v for k, v in scores_rag.items() if isinstance(v, list)}
        for col, values in score_cols_rag.items():
            df[f"{col}_rag"] = values

    results_path = model_dir / "results.parquet"
    df.to_parquet(results_path, index=False)
    print(f"  Saved {results_path} ({len(df)} rows)")

    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Next step: python 6.1-oos-label.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()