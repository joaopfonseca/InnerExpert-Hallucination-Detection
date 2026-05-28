"""Run all hallucination detection methods on OOD test data and save predictions.

This script is the HEADLINE EVALUATION of the project. It runs inference
with every method (our trained detector, all baselines, HaluNet) on the
held-out 2026 RealtimeQA test data and saves raw prediction scores.

Methods evaluated:
  - PredictiveEntropy (per-token + answer-level via mean/max aggregation)
  - LLM-Check (attention, hidden, perplexity, entropy — per-token + answer-level)
  - SemanticUncertainty (answer-level, requires multi-sample data)
  - SemanticEnergy (answer-level, requires multi-sample data)
  - SelfCheckGPT (NLI + Prompt — answer-level, requires multi-sample data)
  - HaluNet (answer-level, requires log_likelihoods/entropies/embeddings)
  - Our MoE detector (per-token, from detector.pkl)

Saved models (from 3.X scripts):
  - models/<model_slug>/detector.pkl
  - models/<model_slug>/halunet.pt
  - models/<model_slug>/thresholds.json

This script requires model outputs saved by 1.0-generate-answers.py.

**Dependencies on batch file keys:**
  - PredictiveEntropy: requires `scores` (raw output logits, shape B×seq×vocab).
    If unavailable, falls back to `scores_entropy` (top-k entropy, less ideal).
  - LLM-Check: uses pre-computed `attention_scores`, `hidden_scores`,
    and `scores_entropy` from batch files. `sequences` + `input_ids` for perplexity.
  - HaluNet: requires `log_likelihoods` and `entropies` (return_baseline_features=True).
  - SU/SE/SelfCheckGPT: require multi-sample data via load_sampled_outputs.

Output:
  - data/<dataset_slug>/<model_slug>/predictions/*.parquet (one per method)
  - data/<dataset_slug>/<model_slug>/predictions/ground_truth.parquet

Usage:
    python 4.0-model-evaluation.py --test-years 2026 --test-month 1
    python 4.0-model-evaluation.py --test-years 2026 --test-month 2 --model allenai/OLMoE-1B-7B-0924-Instruct
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))

from moeuncert.experiments import (
    resolve_model_slug,
    resolve_dataset_slug,
    load_multi_year_data,
    create_token_labels,
    find_generation_boundaries,
)

# Short alias for convenience within this script
_find_gen_boundaries = find_generation_boundaries

# Backward-compat alias for pickles trained before _replace_inf_with_nan
# moved to moeuncert.experiments.utils.
from moeuncert.experiments import replace_inf_with_nan
_replace_inf_with_nan = replace_inf_with_nan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_composite_qids(outputs: Dict) -> np.ndarray:
    """Build composite question_id::evidence_present keys to disambiguate
    base vs RAG rows that share the same question_id."""
    raw_qids = np.asarray(outputs["question_id"])

    if "evidence_present" not in outputs:
        ev = np.full(len(raw_qids), False, dtype=bool)
    else:
        ev = np.asarray(outputs["evidence_present"])
        if len(ev) != len(raw_qids):
            raise ValueError(
                "Length mismatch in _build_composite_qids(): "
                f"question_id has {len(raw_qids)} entries but "
                f"evidence_present has {len(ev)}"
            )

    return np.array([f"{q}::{int(e)}" for q, e in zip(raw_qids, ev)])


def _build_label_lookup(df_labeled: pd.DataFrame) -> Dict[str, int]:
    """Build {composite_qid: label} lookup from labeled dataframe.

    Uses LLM label when available, falling back to weak label per row.
    """
    if "label_llm_answer" not in df_labeled.columns and "label_weak_hallucination" not in df_labeled.columns:
        raise ValueError("No hallucination labels found in dataframe")

    llm_labels = df_labeled.get("label_llm_answer", pd.Series(dtype=float))
    weak_labels = df_labeled.get("label_weak_hallucination", pd.Series(dtype=float))
    labels = llm_labels.where(llm_labels.notna(), weak_labels).astype(int).values

    qids = df_labeled["question_id"].astype(str).values
    ev_col = next(
        (c for c in ("evidence_present", "has_evidence", "with_evidence")
         if c in df_labeled.columns),
        None,
    )
    if ev_col is not None:
        keys = np.array([f"{q}::{int(e)}" for q, e in zip(qids, df_labeled[ev_col].values)])
    else:
        keys = qids
    return dict(zip(keys, labels))


def _aggregate_token_to_answer(
    token_scores: np.ndarray,
    question_ids: np.ndarray,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-token scores to answer-level."""
    unique_qids = np.unique(question_ids)
    aggregated = []
    for qid in unique_qids:
        mask = question_ids == qid
        if aggregation == "mean":
            aggregated.append(token_scores[mask].mean())
        elif aggregation == "max":
            aggregated.append(token_scores[mask].max())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return unique_qids, np.array(aggregated)


def _decode_sampled_responses(
    tokenizer, responses_tokens: List[List[int]]
) -> List[str]:
    """Decode sampled response token IDs to strings."""
    return [
        tokenizer.decode(tokens, skip_special_tokens=True)
        for tokens in responses_tokens
    ]


# ---------------------------------------------------------------------------
# Per-method evaluation functions
# ---------------------------------------------------------------------------

def evaluate_predictive_entropy(
    outputs: Dict,
    comp_qids: np.ndarray,
    label_lookup: Dict[str, int],
) -> Dict[str, pd.DataFrame]:
    """Evaluate PredictiveEntropy — per-token and answer-level (mean + max).

    Tries to use raw `scores` tensor for full-vocabulary entropy.
    Falls back to `scores_entropy` (top-k) if raw scores unavailable.
    """
    # Prefer raw scores for full-vocabulary entropy
    if "scores" in outputs:
        scores = outputs["scores"]  # (B, seq_len, vocab)
        probs = torch.softmax(scores, dim=-1)
        entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    elif "scores_entropy" in outputs:
        # Fallback: use pre-computed full-vocabulary entropy
        print("  WARNING: raw scores not in batch files, using scores_entropy fallback.")
        entropies = outputs["scores_entropy"]  # (B, seq_len)
    else:
        raise KeyError("Neither 'scores' nor 'scores_entropy' found in outputs.")

    ent_flat = entropies.numpy().flatten()

    # Per-token
    B, seq_len = entropies.shape
    token_qids = np.repeat(comp_qids, seq_len)
    token_positions = np.tile(np.arange(seq_len), B)

    token_df = pd.DataFrame({
        "question_id": token_qids,
        "token_position": token_positions,
        "score": ent_flat,
    })

    # Answer-level (mean + max)
    results = {"predictive_entropy": token_df}

    for agg in ["mean", "max"]:
        uq, agg_scores = _aggregate_token_to_answer(ent_flat, token_qids, agg)
        mask = np.array([q in label_lookup for q in uq])
        answer_df = pd.DataFrame({
            "question_id": uq[mask],
            "score": agg_scores[mask],
        })
        results[f"predictive_entropy_{agg}"] = answer_df

    return results


def evaluate_llm_check(
    outputs: Dict,
    comp_qids: np.ndarray,
    label_lookup: Dict[str, int],
) -> pd.DataFrame:
    """Evaluate all four LLM-Check score types using pre-computed batch metrics.

    Returns a single DataFrame with columns:
        question_id, token_position, attention_score, hidden_score,
        perplexity_score, entropy_score

    Key shape reference:
      - attention_scores: (B, n_layers, n_heads, seq_len) — cumsum log-det
      - hidden_scores: (B, seq_len, n_layers) — cumsum of SVD singular values
      - scores_entropy: (B, seq_len) — top-k output entropy
      - Perplexity computed from sequences + scores (or scores_entropy fallback)
    """
    B = len(comp_qids)

    # --- Attention ---
    # Use pre-computed attention_scores: (B, n_layers, n_heads, seq_len)
    # Average across heads and layers for per-position score
    if "attention_scores" not in outputs:
        raise KeyError("'attention_scores' not found in outputs.")
    attn_s = outputs["attention_scores"]  # (B, n_layers, n_heads, seq_len)
    # Mean over heads, then mean over layers → (B, seq_len)
    attn_mean = attn_s.mean(dim=1).mean(dim=1).numpy()  # (B, seq_len)

    # --- Hidden ---
    # Use pre-computed hidden_scores: (B, seq_len, n_layers)
    # Mean over layers for per-position score
    if "hidden_scores" not in outputs:
        raise KeyError("'hidden_scores' not found in outputs.")
    hidden_s = outputs["hidden_scores"]  # (B, seq_len, n_layers)
    hidden_mean = hidden_s.mean(dim=-1).numpy()  # (B, seq_len)

    # --- Perplexity ---
    # Use pre-computed perplexity if available (from compute_metrics with
    # return_baseline_features=True), otherwise recompute from raw scores
    # or fall back to NaN.
    sequences = outputs["sequences"]
    input_ids = outputs["input_ids"]

    perplexities = np.full(B, np.nan)
    if "perplexity" in outputs:
        perplexities = outputs["perplexity"].numpy()  # (B,)
    elif "scores" in outputs:
        scores_t = outputs["scores"]  # (B, seq_len, vocab)
        for b in range(B):
            gen_start, gen_end = _find_gen_boundaries(input_ids[b], sequences[b])
            gen_len = gen_end - gen_start
            if gen_len > 0:
                log_probs = torch.log_softmax(scores_t[b, :gen_len], dim=-1)
                gen_tokens = sequences[b, gen_start:gen_end].unsqueeze(-1)
                token_logprobs = torch.gather(log_probs, dim=-1, index=gen_tokens).squeeze(-1)
                perplexities[b] = torch.exp(-token_logprobs.mean()).item()
    else:
        print("  WARNING: perplexity not available (requires return_baseline_features=True or raw scores).")

    # --- Entropy ---
    # Use scores_entropy (top-k entropy) or recompute from raw scores
    if "scores" in outputs:
        scores_t = outputs["scores"]
        probs_t = torch.softmax(scores_t, dim=-1)
        entropy_scores = -(probs_t * torch.log(probs_t + 1e-10)).sum(dim=-1)
    elif "scores_entropy" in outputs:
        entropy_scores = outputs["scores_entropy"]  # (B, seq_len)
    else:
        raise KeyError("Neither 'scores' nor 'scores_entropy' found.")

    seq_len_out = entropy_scores.shape[1]

    rows = []
    for b in range(B):
        qid = comp_qids[b]
        for pos in range(seq_len_out):
            rows.append({
                "question_id": qid,
                "token_position": pos,
                "attention_score": float(attn_mean[b, pos]),
                "hidden_score": float(hidden_mean[b, pos]),
                "perplexity_score": float(perplexities[b]),
                "entropy_score": float(entropy_scores[b, pos].item()
                    if isinstance(entropy_scores, torch.Tensor)
                    else entropy_scores[b, pos]),
            })

    return pd.DataFrame(rows)


def evaluate_semantic_uncertainty(
    model: str,
    test_years: List[int],
    test_month: Optional[int],
    data_root: Path,
    num_samples: int = 5,
) -> pd.DataFrame:
    """Evaluate SemanticUncertainty on sampled test data.

    Needs to decode token IDs to strings for NLI clustering.
    """
    from moeuncert.experiments.data_loading import load_sampled_outputs
    from moeuncert.baselines.semantic_uncertainty import SemanticUncertainty
    from transformers import AutoTokenizer

    sampled = load_sampled_outputs(
        data_root, test_years, test_month, model, num_samples=num_samples
    )
    su = SemanticUncertainty()
    tokenizer = AutoTokenizer.from_pretrained(model)

    rows = []
    for qid, responses_tokens in sampled["responses_by_qid"].items():
        logprobs = sampled["log_probs_by_qid"].get(qid, [])
        if not responses_tokens or len(responses_tokens) < 2:
            continue
        responses = _decode_sampled_responses(tokenizer, responses_tokens)
        try:
            score = su.predict_proba(responses, logprobs)
            rows.append({"question_id": qid, "score": float(score)})
        except Exception as e:
            print(f"  WARNING: Failed SU for qid={qid}: {e}")

    return pd.DataFrame(rows)


def evaluate_semantic_energy(
    model: str,
    test_years: List[int],
    test_month: Optional[int],
    data_root: Path,
    num_samples: int = 5,
) -> pd.DataFrame:
    """Evaluate SemanticEnergy on sampled test data.

    Needs to decode token IDs to text for NLI clustering to get cluster
    assignments, then compute SemanticEnergy scores per cluster.
    Mirrors the approach in 3.1-fit-baselines.py.
    """
    from moeuncert.experiments.data_loading import load_sampled_outputs
    from moeuncert.baselines.semantic_energy import SemanticEnergy
    from moeuncert.baselines.semantic_uncertainty import (
        SemanticUncertainty,
        semantic_ids_to_clusters,
    )
    from transformers import AutoTokenizer

    sampled = load_sampled_outputs(
        data_root, test_years, test_month, model, num_samples=num_samples
    )
    su = SemanticUncertainty()
    se = SemanticEnergy()
    tokenizer = AutoTokenizer.from_pretrained(model)

    rows = []
    for qid, responses_tokens in sampled["responses_by_qid"].items():
        logprobs = sampled["log_probs_by_qid"].get(qid, [])
        response_logits = sampled["logits_by_qid"].get(qid, [])

        if not responses_tokens or not response_logits or len(responses_tokens) < 2:
            continue

        try:
            responses = _decode_sampled_responses(tokenizer, responses_tokens)
            # Cluster responses via NLI
            semantic_ids = su.cluster_responses(responses)
            clusters = semantic_ids_to_clusters(semantic_ids)

            # Compute per-response probabilities
            response_probs = [[np.exp(lp) for lp in ll] for ll in logprobs]

            score = se.predict_proba(
                response_logits=response_logits,
                response_probs=response_probs,
                clusters=clusters,
            )
            rows.append({"question_id": qid, "score": float(score)})
        except Exception as e:
            print(f"  WARNING: Failed SEnergy for qid={qid}: {e}")
            continue

    return pd.DataFrame(rows)


def evaluate_selfcheck(
    model: str,
    test_years: List[int],
    test_month: Optional[int],
    data_root: Path,
    variant: str = "nli",
    num_samples: int = 5,
) -> pd.DataFrame:
    """Evaluate SelfCheckGPT (NLI or Prompt variant) on sampled test data.

    For answer-level evaluation, we pass the entire generated answer as a
    single "sentence". The sampled passages are the multi-sample responses.
    Token IDs are decoded to strings before passing to SelfCheckGPT.
    """
    from moeuncert.experiments.data_loading import load_sampled_outputs
    from moeuncert.baselines import SelfCheckNLI, SelfCheckPrompt
    from transformers import AutoTokenizer

    sampled = load_sampled_outputs(
        data_root, test_years, test_month, model, num_samples=num_samples
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    if variant == "nli":
        checker = SelfCheckNLI()
    elif variant == "prompt":
        checker = SelfCheckPrompt()
    else:
        raise ValueError(f"Unknown SelfCheck variant: {variant}")

    rows = []
    for qid, responses_tokens in sampled["responses_by_qid"].items():
        if not responses_tokens or len(responses_tokens) < 2:
            continue
        # First response is the target answer, rest are sampled passages
        responses = _decode_sampled_responses(tokenizer, responses_tokens)
        target = responses[0]
        sampled_passages = responses[1:]

        try:
            scores = checker.predict_proba([target], sampled_passages)
            rows.append({"question_id": qid, "score": float(scores.mean())})
        except Exception as e:
            print(f"  WARNING: Failed selfcheck_{variant} for qid={qid}: {e}")
            continue

    return pd.DataFrame(rows)


def evaluate_halunet(
    outputs: Dict,
    df_labeled: pd.DataFrame,
    halunet_path: Path,
) -> pd.DataFrame:
    """Evaluate HaluNet on test data (single-generation outputs).

    Requires batch files saved with return_baseline_features=True
    (provides log_likelihoods and entropies) PLUS raw hidden_states.
    """
    from moeuncert.baselines.halunet import HaluNet
    from collections import defaultdict

    # HaluNet requires last_hidden_states for embeddings — may not be in batch files
    if "last_hidden_states" not in outputs:
        raise KeyError(
            "'last_hidden_states' not in outputs. "
            "Ensure 1.0-generate-answers.py was run with --return-baseline-features."
        )

    # Load HaluNet checkpoint
    halunet = HaluNet()
    halunet.load(halunet_path)

    # Build qid → index map
    all_qids = outputs["question_id"]
    qid_to_indices = defaultdict(list)
    for i, qid in enumerate(all_qids):
        qid_to_indices[str(qid)].append(i)

    rows = []
    for _, row in df_labeled.iterrows():
        qid = str(row["question_id"])
        if qid not in qid_to_indices:
            continue

        indices = qid_to_indices[qid]
        evidence_present = bool(row.get("evidence_present", False))
        idx = indices[1] if evidence_present and len(indices) > 1 else indices[0]

        input_ids = outputs["input_ids"][idx]
        sequences = outputs["sequences"][idx]
        gen_start, gen_end = _find_gen_boundaries(input_ids, sequences)
        gen_len = gen_end - gen_start
        if gen_len <= 0:
            continue

        # Extract features for generated tokens (pre-sliced to last layer + generated positions)
        ll = outputs["log_likelihoods"][idx, :gen_len].numpy()
        ent = outputs["entropies"][idx, :gen_len].numpy()
        hidden_states = outputs["last_hidden_states"][idx, :gen_len, :].numpy()

        score = halunet.predict_proba(ll, ent, hidden_states)

        # Use composite key
        ev_flag = bool(outputs.get("evidence_present", [False] * len(all_qids))[idx])
        comp_qid = f"{qid}::{int(ev_flag)}"
        rows.append({"question_id": comp_qid, "score": float(score)})

    return pd.DataFrame(rows)


def evaluate_detector(
    outputs: Dict,
    df_labeled: pd.DataFrame,
    detector_path: Path,
) -> pd.DataFrame:
    """Evaluate our trained MoE detector on test data."""
    from collections import defaultdict

    # Load detector
    with open(detector_path, "rb") as f:
        detector_data = pickle.load(f)
    model = detector_data["model"]
    feature_names = detector_data["feature_names"]

    # Build qid → index map
    all_qids = outputs["question_id"]
    qid_to_indices = defaultdict(list)
    for i, qid in enumerate(all_qids):
        qid_to_indices[str(qid)].append(i)

    # Build features per token
    feature_keys = [
        "hidden_scores", "attention_scores", "router_entropy",
        "expert_hidden_scores", "expert_similarities", "expert_usage",
    ]
    feature_specs = []
    for feature_name in feature_names:
        matched_key = next(
            (key for key in feature_keys if feature_name.startswith(f"{key}_")),
            None,
        )
        if matched_key is None:
            raise ValueError(
                f"Unsupported detector feature '{feature_name}' in {detector_path}."
            )

        suffix = feature_name[len(matched_key) + 1:]
        if suffix == "":
            raise ValueError(
                f"Invalid detector feature name '{feature_name}': missing column index."
            )
        if not suffix.isdigit():
            raise ValueError(
                f"Invalid detector feature name '{feature_name}': expected '<group>_<index>'."
            )

        feature_specs.append((matched_key, int(suffix), feature_name))

    all_features = []
    all_comp_qids = []
    all_positions = []

    for _, row in df_labeled.iterrows():
        qid = str(row["question_id"])
        if qid not in qid_to_indices:
            continue

        indices = qid_to_indices[qid]
        evidence_present = bool(row.get("evidence_present", False))
        idx = indices[1] if evidence_present and len(indices) > 1 else indices[0]

        input_ids = outputs["input_ids"][idx]
        sequences = outputs["sequences"][idx]
        gen_start, gen_end = _find_gen_boundaries(input_ids, sequences)
        gen_len = gen_end - gen_start
        if gen_len <= 0:
            continue

        # Extract per-token features from pre-computed metrics
        # Batch file shapes:
        #   attention_scores: (B, n_layers, n_heads, seq_len) — cumsum log-det per head
        #   hidden_scores:    (B, seq_len, n_layers) — cumsum SVD singular values
        #   router_entropy:   (B, seq_len, n_layers) — per-layer router entropy
        #   expert_hidden_scores: (B, seq_len, n_layers) or similar
        #   expert_similarities:  (B, seq_len) — scalar per position
        #   expert_usage:     (B, seq_len, n_layers, n_experts) — cumulative usage
        token_feats = {}
        for key in feature_keys:
            if key not in outputs:
                continue
            tensor = outputs[key][idx]  # tensor for this sample
            if key == "attention_scores":
                # (n_layers, n_heads, seq_len) → extract gen positions
                t = tensor[:, :, gen_start - 1 : gen_end - 1]  # (n_layers, n_heads, gen_len)
                t = t.permute(2, 0, 1)  # (gen_len, n_layers, n_heads)
                token_feats[key] = t.reshape(gen_len, -1)
            elif key in ("hidden_scores", "router_entropy"):
                # (seq_len, n_layers) → extract gen positions
                t = tensor[gen_start - 1 : gen_end - 1]  # (gen_len, n_layers)
                token_feats[key] = t
            elif key == "expert_usage":
                # (seq_len, n_layers, n_experts) → extract gen positions
                t = tensor[gen_start - 1 : gen_end - 1]  # (gen_len, n_layers, n_experts)
                token_feats[key] = t.reshape(gen_len, -1)
            elif key == "expert_similarities":
                # (seq_len, n_layers) or (seq_len,) — slice along seq_len like hidden_scores
                t = tensor[gen_start - 1 : gen_end - 1]
                if t.ndim == 1:
                    t = t.reshape(gen_len, 1)
                token_feats[key] = t
            elif key == "expert_hidden_scores":
                # (seq_len, n_layers) or (seq_len,)
                t = tensor[gen_start - 1 : gen_end - 1]
                if t.ndim == 1:
                    t = t.reshape(gen_len, 1)
                token_feats[key] = t
            elif tensor.ndim >= 2:
                t = tensor[gen_start - 1 : gen_end - 1]
                token_feats[key] = t
            else:
                token_feats[key] = tensor.unsqueeze(0).expand(gen_len, -1)

        # Merge into feature vector matching detector feature_names exactly
        token_feat_arrays = {}
        for key, value in token_feats.items():
            arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[0] != gen_len:
                raise ValueError(
                    f"Feature group '{key}' has {arr.shape[0]} rows, expected {gen_len}."
                )
            token_feat_arrays[key] = arr

        X_tokens = np.zeros((gen_len, len(feature_names)), dtype=np.float32)
        for col_idx, (matched_key, feat_idx, feature_name) in enumerate(feature_specs):
            if matched_key not in token_feat_arrays:
                raise ValueError(
                    f"Missing required feature group '{matched_key}' for detector feature '{feature_name}'."
                )

            group_arr = token_feat_arrays[matched_key]
            if feat_idx >= group_arr.shape[1]:
                raise ValueError(
                    f"Feature '{feature_name}' expects index {feat_idx}, "
                    f"but '{matched_key}' provides {group_arr.shape[1]} column(s)."
                )

            X_tokens[:, col_idx] = group_arr[:, feat_idx]

        y_proba = model.predict_proba(X_tokens)[:, 1]

        comp_qid = f"{qid}::{int(evidence_present)}"
        for pos in range(gen_len):
            all_comp_qids.append(comp_qid)
            all_positions.append(pos)
            all_features.append(y_proba[pos])

    return pd.DataFrame({
        "question_id": all_comp_qids,
        "token_position": all_positions,
        "score": all_features,
    })


def build_ground_truth(
    outputs: Dict,
    df_labeled: pd.DataFrame,
    label_lookup: Dict[str, int],
    tokenizer,
) -> pd.DataFrame:
    """Build ground_truth.parquet with token-level and answer-level labels.

    Uses the tokenizer's offset mapping to accurately convert character-level
    hallucinated spans to token-level binary labels.
    """
    from collections import defaultdict

    all_qids = outputs["question_id"]
    qid_to_indices = defaultdict(list)
    for i, qid in enumerate(all_qids):
        qid_to_indices[str(qid)].append(i)

    rows = []
    for _, row in df_labeled.iterrows():
        qid = str(row["question_id"])
        if qid not in qid_to_indices:
            continue
        indices = qid_to_indices[qid]
        evidence_present = bool(row.get("evidence_present", False))
        idx = indices[1] if evidence_present and len(indices) > 1 else indices[0]
        comp_qid = f"{qid}::{int(evidence_present)}"

        answer_label = label_lookup.get(comp_qid, 0)

        input_ids = outputs["input_ids"][idx]
        sequences = outputs["sequences"][idx]
        gen_start, gen_end = _find_gen_boundaries(input_ids, sequences)
        gen_len = gen_end - gen_start
        if gen_len <= 0:
            continue

        # Token-level labels from hallucinated spans via proper offset mapping
        generated_text = row.get("generated_answer", "")
        tokenized = tokenizer(
            generated_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        n_tokens = len(tokenized["input_ids"])
        actual_n_tokens = min(n_tokens, gen_len)

        hallucinated_spans = row.get("llm_hallucinated_spans", None)
        token_labels = create_token_labels(
            generated_text,
            hallucinated_spans,
            tokenized["offset_mapping"],
        )
        token_labels = token_labels[:actual_n_tokens]

        for pos in range(actual_n_tokens):
            rows.append({
                "question_id": comp_qid,
                "token_position": pos,
                "token_label": int(token_labels[pos].item()),
                "answer_label": answer_label,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all hallucination detection methods on OOD test data"
    )
    parser.add_argument(
        "--test-years", type=int, nargs="+", default=[2026],
        help="Test years (default: 2026)",
    )
    parser.add_argument(
        "--test-month", type=int, default=None,
        help="Test month (for single-year test datasets)",
    )
    parser.add_argument(
        "--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--label-model", type=str, default="zai-org/GLM-5.1",
        help="Label model name",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory with saved models",
    )
    parser.add_argument(
        "--skip-sampled", action="store_true",
        help="Skip methods that require multi-sample data (SU, SE, SelfCheckGPT)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sampled responses per question for SU/SE/SelfCheck (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used during generation (default: 0.7)",
    )

    args = parser.parse_args()

    print(f"{'=' * 70}")
    print("4.0 — MODEL EVALUATION (OOD INFERENCE)")
    print(f"{'=' * 70}")

    model_slug = resolve_model_slug(args.model)

    # Load data first; determine the actual source directory afterwards.
    print("\nLoading test data...")
    df_labeled, outputs, source_dir = load_multi_year_data(
        args.data_root, args.test_years, args.test_month,
        args.model, args.label_model,
    )
    print(f"  {len(df_labeled)} labeled rows")
    print(f"  Source data dir: {source_dir}")

    comp_qids = _build_composite_qids(outputs)
    label_lookup = _build_label_lookup(df_labeled)

    # Create predictions dir inside the actual data source, never in a phantom path.
    predictions_dir = source_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Test dataset: {args.test_years}")
    if args.test_month:
        print(f"Test month: {args.test_month:02d}")

    # -----------------------------------------------------------------------
    # PredictiveEntropy
    # -----------------------------------------------------------------------
    print("\n[1/7] PredictiveEntropy ...")
    pe_results = evaluate_predictive_entropy(outputs, comp_qids, label_lookup)
    for name, df in pe_results.items():
        path = predictions_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # LLM-Check
    # -----------------------------------------------------------------------
    print("\n[2/7] LLM-Check ...")
    llm_df = evaluate_llm_check(outputs, comp_qids, label_lookup)
    path = predictions_dir / "llm_check.parquet"
    llm_df.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(llm_df)} rows)")

    # -----------------------------------------------------------------------
    # Our Detector
    # -----------------------------------------------------------------------
    print("\n[3/7] MoE Detector ...")
    detector_path = args.models_dir / model_slug / "detector.pkl"
    if detector_path.exists():
        det_df = evaluate_detector(outputs, df_labeled, detector_path)
        path = predictions_dir / "detector.parquet"
        det_df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(det_df)} rows)")
    else:
        print(f"  SKIPPED — detector.pkl not found at {detector_path}")

    # -----------------------------------------------------------------------
    # HaluNet
    # -----------------------------------------------------------------------
    print("\n[4/7] HaluNet ...")
    halunet_path = args.models_dir / model_slug / "halunet.pt"
    if halunet_path.exists() and "log_likelihoods" in outputs:
        try:
            hnet_df = evaluate_halunet(outputs, df_labeled, halunet_path)
            path = predictions_dir / "halunet.parquet"
            hnet_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(hnet_df)} rows)")
        except Exception as e:
            print(f"  SKIPPED — error: {e}")
    else:
        missing = []
        if not halunet_path.exists():
            missing.append(f"halunet.pt at {halunet_path}")
        if "log_likelihoods" not in outputs:
            missing.append("log_likelihoods in outputs")
        print(f"  SKIPPED — missing: {', '.join(missing)}")

    # -----------------------------------------------------------------------
    # Sampled-data methods
    # -----------------------------------------------------------------------
    if args.skip_sampled:
        print("\n[5-7] Skipping sampled-data methods (--skip-sampled)")
    else:
        # SemanticUncertainty
        print("\n[5/7] SemanticUncertainty ...")
        try:
            su_df = evaluate_semantic_uncertainty(
                args.model, args.test_years, args.test_month,
                args.data_root, num_samples=args.num_samples,
            )
            path = predictions_dir / "semantic_uncertainty.parquet"
            su_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(su_df)} rows)")
        except Exception as e:
            print(f"  SKIPPED — error: {e}")

        # SemanticEnergy
        print("\n[6/7] SemanticEnergy ...")
        try:
            se_df = evaluate_semantic_energy(
                args.model, args.test_years, args.test_month,
                args.data_root, num_samples=args.num_samples,
            )
            path = predictions_dir / "semantic_energy.parquet"
            se_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(se_df)} rows)")
        except Exception as e:
            print(f"  SKIPPED — error: {e}")

        # SelfCheckGPT (NLI + Prompt)
        print("\n[7/7] SelfCheckGPT ...")
        for variant in ["nli", "prompt"]:
            try:
                sc_df = evaluate_selfcheck(
                    args.model, args.test_years, args.test_month,
                    args.data_root, variant=variant,
                    num_samples=args.num_samples,
                )
                path = predictions_dir / f"selfcheck_{variant}.parquet"
                sc_df.to_parquet(path, index=False)
                print(f"  Saved {path} ({len(sc_df)} rows)")
            except Exception as e:
                print(f"  selfcheck_{variant} SKIPPED — error: {e}")

    # -----------------------------------------------------------------------
    # Ground Truth
    # -----------------------------------------------------------------------
    print("\n[GT] Building ground truth ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gt_df = build_ground_truth(outputs, df_labeled, label_lookup, tokenizer)
    gt_path = predictions_dir / "ground_truth.parquet"
    gt_df.to_parquet(gt_path, index=False)
    print(f"  Saved {gt_path} ({len(gt_df)} rows)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Predictions saved to: {predictions_dir}")
    print(f"Files: {[f.name for f in sorted(predictions_dir.glob('*.parquet'))]}")
    print("\nNext step: python 5.0-results-analysis.py")


if __name__ == "__main__":
    main()
