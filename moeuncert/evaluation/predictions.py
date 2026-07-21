"""Per-method prediction functions shared by 4.0 (RealtimeQA) and 7.0 (OOS).

Extracted from ``experiments/4.0-model-evaluation.py``.  Each
``evaluate_*`` function takes the loaded model outputs dict (and
labelled dataframe where needed) and returns a ``pd.DataFrame`` of
raw prediction scores, ready to be saved as a ``.parquet`` and later
analysed by the metrics layer.

The sampled-data methods (``evaluate_semantic_uncertainty``,
``evaluate_semantic_energy``, ``evaluate_selfcheck``) accept a
``sampled`` dict plus a ``tokenizer`` so the OOS pipeline (7.0) can
load sampled data from a different directory layout without going
through the RealtimeQA-specific ``load_sampled_outputs`` helper.
For backward compatibility, the RealtimeQA callers can still pass
``test_years`` / ``test_month`` / ``data_root`` and the functions
will call ``load_sampled_outputs`` internally.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from moeuncert.experiments import (
    create_token_labels,
    find_generation_boundaries,
    resolve_cache_dir,
)
from moeuncert.experiments.data_loading import _normalize_question_id_value

_find_gen_boundaries = find_generation_boundaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_composite_qids(outputs: Dict) -> np.ndarray:
    """Build ``question_id::evidence_present`` keys to disambiguate
    base vs RAG rows that share the same question_id."""
    raw_qids = np.asarray(outputs["question_id"])

    if "evidence_present" not in outputs:
        ev = np.full(len(raw_qids), False, dtype=bool)
    else:
        ev = np.asarray(outputs["evidence_present"])
        if len(ev) != len(raw_qids):
            raise ValueError(
                "Length mismatch in build_composite_qids(): "
                f"question_id has {len(raw_qids)} entries but "
                f"evidence_present has {len(ev)}"
            )

    return np.array([f"{q}::{int(e)}" for q, e in zip(raw_qids, ev)])


def build_label_lookup(df_labeled: pd.DataFrame) -> Dict[str, int]:
    """Build ``{composite_qid: label}`` lookup from labeled dataframe.

    Uses LLM label when available, falling back to weak label per row.
    For OOS data without weak labels, only ``label_llm_answer`` is used.
    """
    if (
        "label_llm_answer" not in df_labeled.columns
        and "label_weak_hallucination" not in df_labeled.columns
    ):
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


def aggregate_token_to_answer(
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


def decode_sampled_responses(
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

    Tries to use raw ``scores`` tensor for full-vocabulary entropy.
    Falls back to ``scores_entropy`` (top-k) if raw scores unavailable.
    """
    if "scores" in outputs:
        scores = outputs["scores"]
        probs = torch.softmax(scores, dim=-1)
        entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    elif "scores_entropy" in outputs:
        print("  WARNING: raw scores not in batch files, using scores_entropy fallback.")
        entropies = outputs["scores_entropy"]
    else:
        raise KeyError("Neither 'scores' nor 'scores_entropy' found in outputs.")

    ent_flat = entropies.numpy().flatten()

    B, seq_len = entropies.shape
    token_qids = np.repeat(comp_qids, seq_len)
    token_positions = np.tile(np.arange(seq_len), B)

    token_df = pd.DataFrame({
        "question_id": token_qids,
        "token_position": token_positions,
        "score": ent_flat,
    })

    results = {"predictive_entropy": token_df}

    for agg in ["mean", "max"]:
        uq, agg_scores = aggregate_token_to_answer(ent_flat, token_qids, agg)
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
    """
    B = len(comp_qids)

    if "attention_scores" not in outputs:
        raise KeyError("'attention_scores' not found in outputs.")
    attn_s = outputs["attention_scores"]
    attn_mean = attn_s.mean(dim=1).mean(dim=1).numpy()

    if "hidden_scores" not in outputs:
        raise KeyError("'hidden_scores' not found in outputs.")
    hidden_s = outputs["hidden_scores"]
    hidden_mean = hidden_s.mean(dim=-1).numpy()

    sequences = outputs["sequences"]
    input_ids = outputs["input_ids"]

    perplexities = np.full(B, np.nan)
    if "perplexity" in outputs:
        perplexities = outputs["perplexity"].numpy()
    elif "scores" in outputs:
        scores_t = outputs["scores"]
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

    if "scores" in outputs:
        scores_t = outputs["scores"]
        probs_t = torch.softmax(scores_t, dim=-1)
        entropy_scores = -(probs_t * torch.log(probs_t + 1e-10)).sum(dim=-1)
    elif "scores_entropy" in outputs:
        entropy_scores = outputs["scores_entropy"]
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
                "entropy_score": float(
                    entropy_scores[b, pos].item()
                    if isinstance(entropy_scores, torch.Tensor)
                    else entropy_scores[b, pos]
                ),
            })

    return pd.DataFrame(rows)


def evaluate_semantic_uncertainty(
    model: str,
    test_years: Optional[List[int]] = None,
    test_month: Optional[int] = None,
    data_root: Optional[Path] = None,
    num_samples: int = 5,
    sampled: Optional[Dict] = None,
    tokenizer=None,
    evidence_present: int = 0,
) -> pd.DataFrame:
    """Evaluate SemanticUncertainty on sampled test data.

    Two calling conventions:
    1. RealtimeQA: pass ``model``, ``test_years``, ``test_month``,
       ``data_root``; the function calls ``load_sampled_outputs``
       internally.
    2. OOS / pre-loaded: pass ``sampled`` (the dict from a sampled
       loader) and ``tokenizer`` directly.  This avoids the
       RealtimeQA-specific directory resolution.
    """
    from moeuncert.baselines.semantic_uncertainty import SemanticUncertainty

    if sampled is None:
        from moeuncert.experiments.data_loading import load_sampled_outputs
        sampled = load_sampled_outputs(
            data_root, test_years, test_month, model,
            num_samples=num_samples, with_evidence=bool(evidence_present),
        )

    if tokenizer is None:
        from moeuncert.experiments import load_tokenizer_for_data
        tokenizer = load_tokenizer_for_data(model)

    su = SemanticUncertainty()

    rows = []
    for qid, responses_tokens in sampled["responses_by_qid"].items():
        logprobs = sampled["log_probs_by_qid"].get(qid, [])
        if not responses_tokens or len(responses_tokens) < 2:
            continue
        responses = decode_sampled_responses(tokenizer, responses_tokens)
        try:
            score = su.predict_proba(responses, logprobs)
            clean_qid = _normalize_question_id_value(qid)
            rows.append({
                "question_id": f"{clean_qid}::{evidence_present}",
                "score": float(np.nan_to_num(score, posinf=1e10, neginf=-1e10)),
            })
        except Exception as e:
            print(f"  WARNING: Failed SU for qid={qid}: {e}")

    return pd.DataFrame(rows)


def evaluate_semantic_energy(
    model: str,
    test_years: Optional[List[int]] = None,
    test_month: Optional[int] = None,
    data_root: Optional[Path] = None,
    num_samples: int = 5,
    sampled: Optional[Dict] = None,
    tokenizer=None,
    evidence_present: int = 0,
) -> pd.DataFrame:
    """Evaluate SemanticEnergy on sampled test data.

    Two calling conventions — see ``evaluate_semantic_uncertainty``.
    """
    from moeuncert.baselines.semantic_energy import SemanticEnergy
    from moeuncert.baselines.semantic_uncertainty import (
        SemanticUncertainty,
        semantic_ids_to_clusters,
    )

    if sampled is None:
        from moeuncert.experiments.data_loading import load_sampled_outputs
        sampled = load_sampled_outputs(
            data_root, test_years, test_month, model,
            num_samples=num_samples, with_evidence=bool(evidence_present),
        )

    if tokenizer is None:
        from moeuncert.experiments import load_tokenizer_for_data
        tokenizer = load_tokenizer_for_data(model)

    su = SemanticUncertainty()
    se = SemanticEnergy()

    rows = []
    for qid, responses_tokens in sampled["responses_by_qid"].items():
        logprobs = sampled["log_probs_by_qid"].get(qid, [])
        response_logits = sampled["logits_by_qid"].get(qid, [])

        if not responses_tokens or not response_logits or len(responses_tokens) < 2:
            continue

        try:
            responses = decode_sampled_responses(tokenizer, responses_tokens)
            semantic_ids = su.cluster_responses(responses)
            clusters = semantic_ids_to_clusters(semantic_ids)

            response_probs = [[np.exp(lp) for lp in ll] for ll in logprobs]

            score = se.predict_proba(
                response_logits=response_logits,
                response_probs=response_probs,
                clusters=clusters,
            )
            clean_qid = _normalize_question_id_value(qid)
            rows.append({
                "question_id": f"{clean_qid}::{evidence_present}",
                "score": float(np.nan_to_num(score, posinf=1e10, neginf=-1e10)),
            })
        except Exception as e:
            print(f"  WARNING: Failed SEnergy for qid={qid}: {e}")
            continue

    return pd.DataFrame(rows)


def evaluate_selfcheck(
    model: str,
    test_years: Optional[List[int]] = None,
    test_month: Optional[int] = None,
    data_root: Optional[Path] = None,
    variant: str = "nli",
    num_samples: int = 5,
    sampled: Optional[Dict] = None,
    tokenizer=None,
    evidence_present: int = 0,
) -> pd.DataFrame:
    """Evaluate SelfCheckGPT (NLI or Prompt variant) on sampled test data.

    Two calling conventions — see ``evaluate_semantic_uncertainty``.
    """
    from moeuncert.baselines import SelfCheckNLI, SelfCheckPrompt

    if sampled is None:
        from moeuncert.experiments.data_loading import load_sampled_outputs
        sampled = load_sampled_outputs(
            data_root, test_years, test_month, model,
            num_samples=num_samples, with_evidence=bool(evidence_present),
        )

    if tokenizer is None:
        from moeuncert.experiments import load_tokenizer_for_data
        tokenizer = load_tokenizer_for_data(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if variant == "nli":
        checker = SelfCheckNLI(device=device)
    elif variant == "prompt":
        checker = SelfCheckPrompt(device=device)
    else:
        raise ValueError(f"Unknown SelfCheck variant: {variant}")

    rows = []
    for qid, responses_tokens in sampled["responses_by_qid"].items():
        if not responses_tokens or len(responses_tokens) < 2:
            continue
        responses = decode_sampled_responses(tokenizer, responses_tokens)
        target = responses[0]
        sampled_passages = responses[1:]

        try:
            scores = checker.predict_proba([target], sampled_passages)
            clean_qid = _normalize_question_id_value(qid)
            rows.append({
                "question_id": f"{clean_qid}::{evidence_present}",
                "score": float(np.nan_to_num(scores.mean(), posinf=1.0, neginf=0.0)),
            })
        except Exception as e:
            print(f"  WARNING: Failed selfcheck_{variant} for qid={qid}: {e}")
            continue

    return pd.DataFrame(rows)


def evaluate_halunet(
    outputs: Dict,
    df_labeled: pd.DataFrame,
    halunet_path: Path,
) -> pd.DataFrame:
    """Evaluate HaluNet on test data (single-generation outputs)."""
    from moeuncert.baselines.halunet import HaluNet

    if "last_hidden_states" not in outputs:
        raise KeyError(
            "'last_hidden_states' not in outputs. "
            "Ensure generation was run with --return-baseline-features."
        )

    halunet = HaluNet()
    halunet.load(halunet_path)

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

        ll = outputs["log_likelihoods"][idx, :gen_len].numpy()
        ent = outputs["entropies"][idx, :gen_len].numpy()
        hidden_states = outputs["last_hidden_states"][idx, :gen_len, :].numpy()

        score = halunet.predict_proba(ll, ent, hidden_states)

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
    with open(detector_path, "rb") as f:
        detector_data = pickle.load(f)
    model = detector_data["model"]
    feature_names = detector_data["feature_names"]

    all_qids = outputs["question_id"]
    qid_to_indices = defaultdict(list)
    for i, qid in enumerate(all_qids):
        qid_to_indices[str(qid)].append(i)

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

        token_feats = {}
        for key in feature_keys:
            if key not in outputs:
                continue
            tensor = outputs[key][idx]
            if key == "attention_scores":
                t = tensor[:, :, gen_start - 1 : gen_end - 1]
                t = t.permute(2, 0, 1)
                token_feats[key] = t.reshape(gen_len, -1)
            elif key in ("hidden_scores", "router_entropy"):
                t = tensor[gen_start - 1 : gen_end - 1]
                token_feats[key] = t
            elif key == "expert_usage":
                t = tensor[gen_start - 1 : gen_end - 1]
                token_feats[key] = t.reshape(gen_len, -1)
            elif key == "expert_similarities":
                t = tensor[gen_start - 1 : gen_end - 1]
                if t.ndim == 1:
                    t = t.reshape(gen_len, 1)
                token_feats[key] = t
            elif key == "expert_hidden_scores":
                t = tensor[gen_start - 1 : gen_end - 1]
                if t.ndim == 1:
                    t = t.reshape(gen_len, 1)
                token_feats[key] = t
            elif tensor.ndim >= 2:
                t = tensor[gen_start - 1 : gen_end - 1]
                token_feats[key] = t
            else:
                token_feats[key] = tensor.unsqueeze(0).expand(gen_len, -1)

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