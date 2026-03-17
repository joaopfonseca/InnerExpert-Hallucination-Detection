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
- Optional LLM-based evaluation through the OpenAI API

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

- Optional LLM-extracted hallucinated spans (OpenAI)
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
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.experiments.utils import optimal_threshold

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
METRIC_PREFIXES = ("rouge", "bert", "bleu")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def resolve_dataset_slug(
    years: Optional[Sequence[int]], month: Optional[int]
) -> Tuple[List[int], Optional[int], str]:
    """Resolve dataset period and slug using the same convention as 3.0/3.1."""
    if years is None:
        time_now = datetime.now()
        month = time_now.month - 1 if time_now.month > 1 else 12
        year = time_now.year if time_now.month > 1 else time_now.year - 1
        years = [year]
    else:
        years = list(years)

    if len(years) > 1 and month is not None:
        raise ValueError("Month cannot be specified when multiple years are provided.")

    if len(years) == 1 and month is not None:
        dataset_slug = f"realtimeqa-{years[0]}-{month:02d}"
    else:
        dataset_slug = "realtimeqa-" + "-".join(str(y) for y in years)

    return years, month, dataset_slug


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


def _extract_json_obj(text: str) -> Optional[dict]:
    """Extract first JSON object from model text output."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _openai_chat_json(
    prompt: str, api_key: str, model: str, timeout: int = 60
) -> Optional[dict]:
    """Call OpenAI Chat Completions API and parse JSON payload from response text."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Return concise, valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        return None
    content = resp.json()["choices"][0]["message"]["content"]
    return _extract_json_obj(content)


def generate_llm_labels_and_spans(
    df: pd.DataFrame,
    api_key: Optional[str],
    model: Optional[str],
    max_samples: Optional[int] = None,
    include_spans: bool = True,
) -> Tuple[pd.Series, pd.Series, Dict[int, List[str]]]:
    """
    Generate answer-level labels and token-level hallucinated spans in one API call per sample.

    Returns:
        labels: series with 1 (hallucinated), 0 (grounded), or NaN when unavailable
        reasons: short textual explanation from LLM
        spans_by_row: mapping from row index to extracted hallucinated spans
    """
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    reasons = pd.Series("", index=df.index, dtype=str)
    spans_by_row: Dict[int, List[str]] = {}

    if not api_key or not model:
        return labels, reasons, spans_by_row

    work_index = list(df.index)[:max_samples] if max_samples is not None else list(df.index)
    for idx in work_index:
        row = df.loc[idx]
        prompt = (
            "Given a question, evidence, and model answer, decide whether the answer contains "
            "unsupported or hallucinated content.\n"
            "Return valid JSON only with this exact schema:\n"
            "{\"label\": 1 or 0, "
            "\"hallucinated_spans\": [\"exact substring from answer\", ...]}\n"
            "Rules:\n"
            "- label = 1 means hallucinated/unsupported, label = 0 means grounded/correct.\n"
            "- hallucinated_spans must be exact substrings from the answer.\n"
            "- if label = 0, hallucinated_spans must be an empty list.\n\n"
            f"Question: {row.get('question_sentence', '')}\n"
            f"Evidence: {row.get('evidence', '')}\n"
            f"Answer: {row.get('generated_answer', '')}\n"
        )

        obj = _openai_chat_json(prompt, api_key=api_key, model=model)
        if not obj:
            continue

        label = obj.get("label", None)
        if label in (0, 1):
            labels.loc[idx] = int(label)
        reasons.loc[idx] = str(obj.get("reason", ""))

        if include_spans:
            spans = obj.get("hallucinated_spans", [])
            if isinstance(spans, list):
                spans_by_row[idx] = [str(x) for x in spans if str(x).strip()]
    return labels, reasons, spans_by_row


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Split text into whitespace tokens and return character spans."""
    tokens = []
    spans = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def support_lexicon(*texts: str) -> set:
    """Create normalized token set from supporting texts."""
    lex = set()
    for txt in texts:
        for tok in re.findall(r"\w+", str(txt).lower()):
            norm = _normalize_token(tok)
            if norm:
                lex.add(norm)
    return lex


def mask_from_spans(
    answer: str, token_spans: List[Tuple[int, int]], hallucinated_spans: Sequence[str]
) -> List[int]:
    """Map hallucinated substrings to token mask."""
    ans_lower = answer.lower()
    mark = [0] * len(token_spans)
    for span_text in hallucinated_spans:
        span_text = str(span_text).strip().lower()
        if not span_text:
            continue
        start = ans_lower.find(span_text)
        while start != -1:
            end = start + len(span_text)
            for i, (tok_s, tok_e) in enumerate(token_spans):
                if tok_e > start and tok_s < end:
                    mark[i] = 1
            start = ans_lower.find(span_text, start + 1)
    return mark


def heuristic_token_mask(
    answer: str, answer_label: int, evidence: str, reference: str
) -> Tuple[List[str], List[int]]:
    """Create token hallucination mask using lexical support heuristic."""
    tokens, _ = tokenize_with_spans(answer)
    if answer_label == 0 or not tokens:
        return tokens, [0] * len(tokens)

    support = support_lexicon(evidence, reference)
    mask = []
    for token in tokens:
        norm = _normalize_token(token)
        if not norm or norm in STOPWORDS or len(norm) <= 2:
            mask.append(0)
        else:
            mask.append(0 if norm in support else 1)
    return tokens, mask


if __name__ == "__main__":
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
        "--openai-api-key",
        type=str,
        default=None,
        help="Optional OpenAI API key for LLM-based labels. If not set, it will search the OPENAI_API_KEY environment variable. If still not found, LLM labeling will be skipped.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5.4",
        help="Optional OpenAI model for LLM-based labels. Defaults to 'gpt-5.4'.",
    )
    parser.add_argument(
        "--max-llm-answer-samples",
        type=int,
        default=None,
        help="Maximum rows to query for answer-level LLM labels (optional).",
    )
    parser.add_argument(
        "--use-openai-token-labels",
        action="store_true",
        help="If set, request token-level hallucinated spans from OpenAI for hallucinated answers.",
    )
    parser.add_argument(
        "--max-llm-token-samples",
        type=int,
        default=None,
        help="Maximum rows to query for token-level LLM spans (optional).",
    )
    parser.add_argument(
        "--min-llm-train-samples",
        type=int,
        default=10,
        help="Minimum labeled rows required to train answer-level classifier on LLM labels.",
    )
    args = parser.parse_args()
    load_dotenv()

    if args.openai_api_key is None:
        args.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if args.openai_api_key is None:
        print("OpenAI API key not found. LLM-based labeling will be skipped.")

    model_slug = args.model.replace("/", "__")
    _, _, dataset_slug = resolve_dataset_slug(args.years, args.month)

    data_dir = Path("data") / dataset_slug / model_slug
    input_path = data_dir / args.input_file
    output_path = data_dir / args.output_file
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


    raise NotImplementedError("Checkpoint")


    llm_labels, llm_reasons, llm_spans_from_joint_call = generate_llm_labels_and_spans(
        df,
        api_key=args.openai_api_key,
        model=args.openai_model,
        max_samples=args.max_llm_answer_samples,
        include_spans=args.use_openai_token_labels,
    )
    df["label_llm_answer"] = llm_labels
    df["label_llm_reason"] = llm_reasons

    train_mask = df["label_llm_answer"].notna()
    source = "weak_heuristic"
    if (
        train_mask.sum() >= args.min_llm_train_samples
        and df.loc[train_mask, "label_llm_answer"].nunique() > 1
    ):
        scaler = StandardScaler()
        x_all = answer_feats[clf_features].to_numpy()
        x_train = answer_feats.loc[train_mask, clf_features].to_numpy()
        y_train = df.loc[train_mask, "label_llm_answer"].astype(int).to_numpy()

        x_train_scaled = scaler.fit_transform(x_train)
        x_all_scaled = scaler.transform(x_all)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train_scaled, y_train)
        hallucination_confidence = clf.predict_proba(x_all_scaled)[:, 1]
        source = "llm_classifier"
    else:
        hallucination_confidence = answer_feats["weak_hallucination_score"].to_numpy()

    df["label_hallucination_confidence"] = hallucination_confidence
    df["label_hallucinated_answer"] = (
        df["label_hallucination_confidence"] >= args.answer_threshold
    ).astype(int)
    df["label_answer_source"] = source

    for col, info in thresholds.items():
        df[f"threshold_{col}"] = info["threshold"]
        df[f"threshold_dir_{col}"] = info["direction"]
        df[f"threshold_acc_{col}"] = info["accuracy"]

    # Generate token-level labels
    llm_token_spans = {}
    if args.use_openai_token_labels:
        llm_token_spans = llm_spans_from_joint_call
        if args.max_llm_token_samples is not None:
            # Optional cap for token-level labeling use, while still doing a single API pass.
            keep_ids = set(list(df.index[df.index.isin(llm_token_spans.keys())])[:args.max_llm_token_samples])
            llm_token_spans = {k: v for k, v in llm_token_spans.items() if k in keep_ids}

    answer_tokens_col = []
    token_mask_col = []
    token_ratio_col = []
    token_source_col = []
    llm_spans_col = []

    for idx, row in df.iterrows():
        answer = str(row.get("generated_answer", ""))
        answer_label = int(row.get("label_hallucinated_answer", 0))
        evidence = str(row.get("evidence", ""))
        reference = str(row.get("answer_str", ""))

        spans = llm_token_spans.get(idx, [])
        tokens, token_spans = tokenize_with_spans(answer)
        if answer_label == 0:
            mask = [0] * len(tokens)
            token_source = "none"
        elif spans:
            mask = mask_from_spans(answer, token_spans, spans)
            if sum(mask) == 0:
                tokens, mask = heuristic_token_mask(answer, answer_label, evidence, reference)
                token_source = "heuristic_fallback"
            else:
                token_source = "llm"
        else:
            tokens, mask = heuristic_token_mask(answer, answer_label, evidence, reference)
            token_source = "heuristic"

        ratio = float(sum(mask) / len(mask)) if mask else 0.0
        answer_tokens_col.append(tokens)
        token_mask_col.append(mask)
        token_ratio_col.append(ratio)
        token_source_col.append(token_source)
        llm_spans_col.append(spans)

    df["answer_tokens"] = answer_tokens_col
    df["label_token_hallucination_mask"] = token_mask_col
    df["label_token_hallucination_ratio"] = token_ratio_col
    df["label_token_source"] = token_source_col
    df["label_llm_token_spans"] = llm_spans_col

    # Save dataset with labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Labeled dataset saved to {output_path}")
    print(
        "Summary:",
        {
            "rows": len(df),
            "answer_hallucination_rate": float(df["label_hallucinated_answer"].mean()),
            "avg_token_hallucination_ratio": float(df["label_token_hallucination_ratio"].mean()),
            "answer_label_source": source,
        },
    )
