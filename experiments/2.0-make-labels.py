"""
Create answer-level and token-level hallucination labels for RealtimeQA outputs.

This script reads the `results.parquet` generated in step 3.0 and produces a
labeled parquet file with additional supervision columns for training
hallucination detection models.

Answer-level labels
-------------------
Answer-level labeling combines:

- Evidence presence (binary)
- BERTScore metrics (`bert_precision`, `bert_recall`, `bert_f1`)
- ROUGE metrics (`rouge1`, `rouge2`, `rougeL`, `rougeLsum`)
- BLEU
- Optional LLM-based evaluation through DeepInfra

For each non-binary metric, thresholds are estimated from base vs RAG
separability (same principle used in `3.1-analyze-metrics.py`, via ROC-derived
accuracy-optimal thresholds). These metric-level hallucination flags are then
combined with evidence presence to create weak answer-level labels.

When OpenAI labels are provided, the a logistic regression classifier is trained
on those labels (if enough labeled samples exist) and output a
model-based hallucination confidence score. Otherwise, it falls back to the
weak heuristic confidence.

Token-level labels
------------------
Token-level labels are stored as masks aligned with answer tokens
(`1 = hallucinated`, `0 = grounded`). Two modes are supported:

- Optional LLM-extracted hallucinated spans (DeepInfra)
- Deterministic lexical-support heuristic fallback using evidence/reference text

If LLM spans are requested but unusable for a sample, the script automatically
falls back to the heuristic method.

Outputs
-------
The output parquet keeps the original rows and appends label columns, including:

- `label_hallucination_confidence`
- `label_hallucinated_answer`
- `label_token_hallucination_mask`
- `label_token_hallucination_ratio`
- `label_answer_source` / `label_token_source`

Dataset/model paths follow the same `--model`, `--years`, `--month` resolution
as scripts 3.0 and 3.1.
"""

import argparse
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.experiments import optimal_threshold, resolve_model_slug, resolve_dataset_slug

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
METRIC_PREFIXES = ("rouge", "bert", "bleu")


def expand_base_rag_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand rows with both base and RAG metrics into separate rows for each, with an 
    evidence_present flag.
    """
    RAG_COLS = df.columns[df.columns.str.endswith("_rag")]
    NON_RAG_COLS = RAG_COLS.str.replace("_rag", "")

    df_base = df.loc[:, ~df.columns.str.endswith("_rag")].copy()
    CORE_COLS = df_base.columns[~df_base.columns.isin(NON_RAG_COLS)]
    df_base["evidence_present"] = 0

    df_rag = df.loc[:, [*CORE_COLS, *RAG_COLS]].copy().rename(
        columns={col: new for col, new in zip(RAG_COLS, NON_RAG_COLS)}
    )
    df_rag["evidence_present"] = 1
    df = pd.concat([df_base, df_rag], ignore_index=True)
    return df


def compute_metric_thresholds(
    df: pd.DataFrame, metric_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Compute optimal thresholds for each metric based on base vs RAG separability."""
    y_true = df["evidence_present"].to_numpy()
    thresholds = {}
    for col in metric_cols:
        scores = df[col].to_numpy()
        threshold, acc = optimal_threshold(y_true, scores)
        direction = "ge" if scores[y_true == 1].mean() >= scores[y_true == 0].mean() else "lt"
        thresholds[col] = {"threshold": threshold, "direction": direction, "accuracy": acc}
    return thresholds


def metric_hallucination_flag(values: pd.Series, threshold: float, direction: str) -> pd.Series:
    """
    Convert a metric into a binary hallucination flag (1=hallucination, 0=grounded).
    """
    if direction == "ge":
        grounded = values >= threshold
    else:
        grounded = values < threshold
    return (~grounded).astype(int)


def build_answer_feature_frame(
    df: pd.DataFrame, metric_cols: Sequence[str], thresholds: Dict[str, Dict[str, float]]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build answer-level feature frame and weak labels.
    """
    feats = pd.DataFrame(index=df.index)
    feats["evidence_present"] = df["evidence_present"]

    feature_cols = ["evidence_present"]
    halluc_flag_cols = []
    for col in metric_cols:
        feats[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        feature_cols.append(col)
        t = thresholds[col]["threshold"]
        d = thresholds[col]["direction"]
        flag_col = f"{col}_halluc_flag"
        feats[flag_col] = metric_hallucination_flag(feats[col], t, d)
        halluc_flag_cols.append(flag_col)

    if halluc_flag_cols:
        feats["weak_hallucination_score"] = feats[halluc_flag_cols].mean(axis=1)
        feats["weak_hallucination_label"] = (
            feats["weak_hallucination_score"] >= 0.5
        ).astype(int)
    return feats, feature_cols


PROMPT_TEMPLATE = (
    "Given a question, evidence, and model answer, decide whether the answer contains "
    "unsupported or hallucinated content.\n"
    "Return valid JSON only with this exact schema:\n"
    '{"label": 1 or 0, '
    '"hallucinated_spans": ["exact substring from answer", ...]}\n'
    "Rules:\n"
    "- label = 1 means hallucinated/unsupported, label = 0 means grounded/correct.\n"
    "- hallucinated_spans must be exact substrings from the answer.\n"
    "- if label = 0, hallucinated_spans must be an empty list.\n\n"
    "Question: {question}\n"
    "Evidence: {evidence}\n"
    "Answer: {answer}\n"
)


def _build_batch_requests(
    df: pd.DataFrame,
    model: str,
    work_index: List[int],
) -> List[str]:
    """Build JSONL lines for the OpenAI Batch API, one per row."""
    lines = []
    for idx in work_index:
        row = df.loc[idx]
        prompt = PROMPT_TEMPLATE.format(
            question=row.get("question_sentence", ""),
            evidence=row.get("evidence", ""),
            answer=row.get("generated_answer", ""),
        )
        request = {
            "custom_id": f"row-{idx}",
            "method": "post",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return concise, valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        }
        lines.append(json.dumps(request))
    return lines


def _parse_batch_response_line(
    line: str,
) -> Tuple[Optional[int], Optional[int], List[str]]:
    """Parse a single line from a batch result JSONL file.

    Returns (row_index, label, spans) where row_index comes from custom_id,
    label is 0/1 or None, and spans is a list of hallucinated substring texts.
    """
    try:
        result = json.loads(line)
    except json.JSONDecodeError:
        return None, None, []

    custom_id = result.get("custom_id", "")
    idx_match = re.search(r"row-(\d+)", custom_id)
    idx = int(idx_match.group(1)) if idx_match else None

    if result.get("error"):
        return idx, None, []

    response = result.get("response", {})
    body = response.get("body", {})
    choices = body.get("choices", [])
    if not choices:
        return idx, None, []

    content = choices[0].get("message", {}).get("content", "").strip()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return idx, None, []

    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return idx, None, []

    label_obj = obj.get("label", None)
    label: Optional[int] = int(label_obj) if label_obj in (0, 1) else None
    spans: List[str] = []
    spans_obj = obj.get("hallucinated_spans", [])
    if isinstance(spans_obj, list):
        spans = [str(x) for x in spans_obj if str(x).strip()]

    return idx, label, spans


def generate_llm_labels_and_spans(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    max_samples: Optional[int] = None,
    poll_interval: int = 10,
) -> Tuple[pd.Series, Dict[int, List[str]]]:
    """
    Generate answer-level labels and token-level hallucinated spans via DeepInfra Batch API.

    Instead of making one API call per sample sequentially, this submits all requests
    as a single batch job, polls until completion, and parses the results. This is
    dramatically faster for large datasets since all requests are processed in parallel
    on the server side.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns question_sentence, evidence, generated_answer.
    client : OpenAI
        OpenAI client configured with DeepInfra base_url and api_key.
    model : str
        Model name on DeepInfra (e.g. "zai-org/GLM-5.1").
    max_samples : int, optional
        Maximum number of rows to label. None means all rows.
    poll_interval : int
        Seconds to wait between batch status checks (default: 10).

    Returns
    -------
    labels : pd.Series
        Series with 1 (hallucinated), 0 (grounded), or NaN when unavailable.
    spans_by_row : dict
        Mapping from row index to list of hallucinated span strings.
    """
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    spans_by_row: Dict[int, List[str]] = {}

    if client is None or not model:
        return labels, spans_by_row

    work_index = list(df.index)[:max_samples] if max_samples is not None else list(df.index)
    if not work_index:
        return labels, spans_by_row

    print(f"Submitting batch of {len(work_index)} requests to DeepInfra...")

    # Build batch request JSONL
    request_lines = _build_batch_requests(df, model, work_index)
    batch_file_content = "\n".join(request_lines)

    # Upload the request file
    uploaded_file = client.files.create(
        file=io.BytesIO(batch_file_content.encode("utf-8")),
        purpose="batch",
    )
    print(f"  Uploaded request file: {uploaded_file.id}")

    # Create the batch job
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Created batch job: {batch.id}")

    # Poll until completed, in_progress, failed, expired, or cancelled
    status = batch.status
    while status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        status = batch.status
        completed = getattr(batch, "request_counts", None)
        total = len(work_index)
        done = completed.completed if completed else "?"
        failed = completed.failed if completed else "?"
        print(f"  Batch {batch.id}: status={status}, completed={done}, failed={failed}, total={total}")

    if status != "completed":
        print(f"  Batch {batch.id} ended with status: {status}")
        if hasattr(batch, "error_file_id") and batch.error_file_id:
            error_content = client.files.content(batch.error_file_id).text
            print(f"  Errors:\n{error_content[:500]}")
        return labels, spans_by_row

    print(f"  Batch {batch.id} completed. Downloading results...")

    # Download and parse results
    output_content = client.files.content(batch.output_file_id).text
    n_parsed = 0
    for line in output_content.strip().split("\n"):
        if not line.strip():
            continue
        idx, label, spans = _parse_batch_response_line(line)
        if idx is not None and idx in labels.index:
            if label is not None:
                labels.loc[idx] = label
            if spans:
                spans_by_row[idx] = spans
            n_parsed += 1

    print(f"  Parsed {n_parsed}/{len(work_index)} results successfully.")

    return labels, spans_by_row


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create answer-level and token-level hallucination labels."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model name.")
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
        "--input-file",
        type=str,
        default="results.parquet",
        help="Input file in data/<dataset_slug>/<model_slug>/.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results_labeled.parquet",
        help="Output labeled file in data/<dataset_slug>/<model_slug>/.",
    )
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=0.5,
        help="Threshold for converting hallucination confidence into binary answer label.",
    )
    parser.add_argument(
        "--deepinfra-model",
        type=str,
        default="zai-org/GLM-5.1",
        help="Optional DeepInfra model used for LLM-based labels (e.g. zai-org/GLM-5.1).",
    )
    parser.add_argument(
        "--max-llm-answer-samples",
        type=int,
        default=None,
        help="Maximum rows to query for answer-level LLM labels (optional).",
    )
    parser.add_argument(
        "--use-llm-token-labels",
        action="store_true",
        help="If set, request token-level hallucinated spans from the LLM for hallucinated answers.",
    )
    parser.add_argument(
        "--max-llm-token-samples",
        type=int,
        default=None,
        help="Maximum rows to query for token-level LLM spans (optional, after answer pass).",
    )
    parser.add_argument(
        "--llm-poll-interval",
        type=int,
        default=10,
        help="Seconds to wait between batch status checks (default: 10).",
    )
    parser.add_argument(
        "--min-llm-train-samples",
        type=int,
        default=10,
        help="Minimum labeled rows required to train answer-level classifier on LLM labels.",
    )
    args = parser.parse_args()

    api_key = os.getenv("DEEPINFRA_API_KEY")
    llm_client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=api_key,
    ) if api_key and args.deepinfra_model else None

    model_slug = resolve_model_slug(args.model)
    _, _, dataset_slug = resolve_dataset_slug(args.years, args.month)

    data_dir = Path("data") / dataset_slug / model_slug
    input_path = data_dir / args.input_file

    # Include LLM labeler model name in output filename
    output_filename = args.output_file
    if args.deepinfra_model:
        model_slug_label = args.deepinfra_model.replace("/", "__")
        stem = Path(output_filename).stem
        suffix = Path(output_filename).suffix
        output_filename = f"{stem}_{model_slug_label}{suffix}"

    output_path = data_dir / output_filename

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read data (same dataset path convention as previous scripts)
    df = pd.read_parquet(input_path).copy()

    # Break answers with and without evidence:
    df = expand_base_rag_rows(df)

    # Generate answer-level labels
    metric_cols = df.columns[df.columns.str.startswith(METRIC_PREFIXES)].tolist()
    thresholds = compute_metric_thresholds(df, metric_cols)
    answer_feats, clf_features = build_answer_feature_frame(df, metric_cols, thresholds)

    df["label_weak_hallucination_score"] = answer_feats["weak_hallucination_score"]
    df["label_weak_hallucination"] = answer_feats["weak_hallucination_label"]

    llm_labels, llm_spans = generate_llm_labels_and_spans(
        df,
        client=llm_client,
        model=args.deepinfra_model,
        max_samples=args.max_llm_answer_samples,
        poll_interval=args.llm_poll_interval,
    )
    df["label_llm_answer"] = llm_labels
    df["llm_hallucinated_spans"] = df.index.map(llm_spans)

    train_mask = df["label_llm_answer"].notna()

    # Train answer-level LR classifier
    scaler = StandardScaler()
    X_all = answer_feats[clf_features].to_numpy()
    X_train = answer_feats.loc[train_mask, clf_features].to_numpy()
    y_train = df.loc[train_mask, "label_llm_answer"].astype(int).to_numpy()

    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)

    clf = LogisticRegression(max_iter=1000)
    calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
    calibrated_clf.fit(X_train_scaled, y_train)
    hallucination_confidence = calibrated_clf.predict_proba(X_all_scaled)[:, 1]

    df["label_hallucination_confidence"] = hallucination_confidence

    # Save dataset with labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Labeled dataset saved to {output_path}")
