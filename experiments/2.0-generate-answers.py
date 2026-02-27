"""
This script demonstrates how to use the OLMoE-1B-7B model for generating
answers to questions in the RealtimeQA dataset, both with and without evidence
(RAG simulation).
"""

from datetime import datetime
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from moeuncert.datasets import fetch_realtimeqa
from moeuncert.utils import generate_params, standardize_outputs, move_to_device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}")

model_name = "allenai/OLMoE-1B-7B-0924-Instruct"
model_slug = model_name.replace("/", "__")

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

# Load a sample of the RealtimeQA dataset
# time_now = datetime.now()
# month = time_now.month - 1 if time_now.month > 1 else 12
# year = time_now.year if time_now.month > 1 else time_now.year - 1
# df = fetch_realtimeqa(split=year, month=month)
# out_dir = Path("data") / model_slug / f"realtimeqa-{year}-{month:02d}"
# out_dir.mkdir(parents=True, exist_ok=True)

years = [2025, 2026]
df = pd.concat([fetch_realtimeqa(split=year) for year in years])

out_dir = Path("data") / model_slug / ("realtimeqa-"+"-".join([str(y) for y in years]))
out_dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Single question generation example

rand_idx = np.random.choice(len(df))

messages = (
    [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant who provides accurate and very concise answers "
                "to questions about recent events. We are currently in "
                f"{pd.to_datetime(df.iloc[0]['question_date']).strftime('%B %Y')}."
            ),
        },
        {"role": "user", "content": df.iloc[rand_idx]["question_sentence"]},
    ],
)
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# NOTE: output_router_logits must be False for `generate`; This is a known limitation.
# See: https://github.com/huggingface/transformers/issues/30731
gen_params = generate_params(
    inputs,
    tokenizer,
    max_new_tokens=65,
    output_attentions=False,
    output_hidden_states=False,
    output_scores=False,
    output_router_logits=False,
)
outputs = model.generate(**gen_params)

print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])

###############################################################################
# Define some functions that will probably need to be moved later on


def tokenize_function(examples, with_evidence=False):
    """Tokenize questions with chat template."""
    texts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant who provides accurate and"
                        "very concise answers to questions about recent"
                        "events. We are currently in "
                        f"{pd.to_datetime(date).strftime('%B %Y')}."
                    ),
                },
                {
                    "role": "user", 
                    "content": f"Evidence: {ev}\n\nQuestion: {q}" if with_evidence else q,
                },
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for q, ev, date in zip(examples["question_sentence"], examples["evidence"], examples["question_date"])
    ]
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


def run_batch_generation(tokenized_dataset, batch_size=2, save_path=None):
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

def read_and_collate_outputs(file_list, get_keys=None):
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
            pad_value = tokenizer.pad_token_id if key in ("sequences", "input_ids") else 0
            padded = []
            for t in value:
                pad_cfg = []
                for d in range(t.dim() - 1, 0, -1):  # F.pad pads from last dim backwards
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

###############################################################################
# Batch generation for all questions in the dataset

dataset = Dataset.from_pandas(df)

batch_size = 6

tokenized = dataset.map(
    lambda examples: tokenize_function(
        examples, with_evidence=False
    ), 
    batched=True, 
    remove_columns=dataset.column_names
)
tokenized.set_format(type="torch")

base_gen_dir = out_dir / "base_generation"
base_gen_dir.mkdir(parents=True, exist_ok=True)
run_batch_generation(tokenized, batch_size, save_path=base_gen_dir)

# Save model output tensors (sequences, hidden_states, attentions, scores, etc.)
# torch.save(all_outputs, out_dir / "model_outputs.pt")

print(f"Model outputs saved to {base_gen_dir}")

#########################################################################################
# Batch generation with evidence (RAG simulation)

tokenized_rag = dataset.map(
    lambda examples: tokenize_function(
        examples, with_evidence=True
    ), 
    batched=True, 
    remove_columns=dataset.column_names
)
tokenized_rag.set_format(type="torch")

evidence_gen_dir = out_dir / "evidence_generation"
evidence_gen_dir.mkdir(parents=True, exist_ok=True)
run_batch_generation(tokenized_rag, batch_size, save_path=evidence_gen_dir)

# Save RAG model output tensors
# torch.save(all_outputs_rag, out_dir / "model_outputs_rag.pt")

print(f"Evidence-based outputs saved to {evidence_gen_dir}")

#########################################################################################
# Metrics
# pip install evaluate rouge_score

import evaluate

all_outputs = read_and_collate_outputs(
    sorted(base_gen_dir.iterdir(), key=lambda p: int(p.stem.split("batch_")[1].split("_of_")[0])),
    get_keys=["generated_answer"]
)
df["generated_answer"] = all_outputs["generated_answer"]

all_outputs_rag = read_and_collate_outputs(
    sorted(evidence_gen_dir.iterdir(), key=lambda p: int(p.stem.split("batch_")[1].split("_of_")[0])),
    get_keys=["generated_answer"]
)
df["generated_answer_rag"] = all_outputs_rag["generated_answer"]


bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")
bleu = evaluate.load("evaluate-metric/bleu")

def compute_scores(candidates, references):
    rouge_scores = rouge.compute(
        predictions=candidates, references=references, use_aggregator=False
    )
    bert_scores = bertscore.compute(
        predictions=candidates, references=references, lang="en", verbose=True
    )
    bleu_scores = [
        bleu.compute(predictions=[candidate], references=[reference])["bleu"]
        if candidate and (reference if isinstance(reference, str) else all(reference))
        else np.nan
        for candidate, reference in zip(candidates, references)
    ]
    return {
        **rouge_scores,
        **{f"bert_{key}": value for key, value in bert_scores.items() if key != "hashcode"},
        "bleu": bleu_scores,
    }

# Sometimes there is no evidence
references = [
    [ev for ev in (gt, evidence) if len(ev.strip()) > 0]
    for gt, evidence in zip(df["answer_str"], df["evidence"])
]
candidates = all_outputs["generated_answer"]
candidates_rag = all_outputs_rag["generated_answer"]

scores = compute_scores(candidates, references)

# Add per-sample scores to the dataframe
score_cols = {k: v for k, v in scores.items() if isinstance(v, list)}
for col, values in score_cols.items():
    df[col] = values

# Compute and store metrics for RAG version
scores_rag = compute_scores(candidates_rag, references)

score_cols_rag = {k: v for k, v in scores_rag.items() if isinstance(v, list)}
for col, values in score_cols_rag.items():
    df[f"{col}_rag"] = values

# Save the dataframe (questions, answers, and per-sample scores) as Parquet
df.to_parquet(out_dir / "results.parquet", index=False)
print(f"DataFrame saved to {out_dir / 'results.parquet'}")
