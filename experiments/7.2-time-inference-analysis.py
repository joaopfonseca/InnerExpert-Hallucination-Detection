"""7.2 — Inference time & memory benchmark for all detection methods.

Measures end-to-end inference time (per 100 displayed tokens) and peak GPU
memory for every hallucination detection method in the pipeline, including
vanilla generation (no detection).

Each method is timed with exactly the instrumentation it needs:
  - gen_time  = wall-clock for model.generate() (the cost of producing the
    displayed answer; for sampling methods, ALL N sampling passes).
  - score_time = wall-clock for post-generation processing (standardize
    outputs, compute metrics, classifier inference, NLI/LLM scoring, etc.).
  - total_time  = gen_time + score_time.
  - time_per_100_tokens = total_time * 100 / total_tokens_displayed.

For sampling-based methods, total_tokens counts only sample 0 (the answer
the user sees); the remaining N-1 background samples contribute to total_time
but not to the displayed-token count.

Data: 50 random RealTimeQA questions from June 2026.
Models: OLMoE-1B-7B, Gemma-4-26B (4-bit quantized).
Output: data/inference-benchmark/{model_slug}/benchmark.csv

Usage:
    python experiments/7.2-time-inference-analysis.py
    python experiments/7.2-time-inference-analysis.py --models allenai/OLMoE-1B-7B-0924-Instruct
"""

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets import fetch_realtimeqa
from moeuncert.utils import standardize_outputs, move_to_device, tokenize_realtimeqa
from moeuncert.monitoring import MoEMonitor
from moeuncert.metrics import compute_metrics
from moeuncert.metrics._metrics import (
    compute_baseline_features,
    topk_entropy,
    hidden_score as _hidden_score,
    attention_score as _attention_score,
    expert_hidden_score as _expert_hidden_score,
    expert_similarity_score as _expert_similarity_score,
    expert_usage_frequency as _expert_usage_frequency,
    expert_usage_gini_impurity as _expert_usage_gini_impurity,
    inverse_herfindahl_index as _inverse_herfindahl_index,
)
from moeuncert.experiments import (
    resolve_model_slug,
    resolve_cache_dir,
    get_quantization_kwargs,
    find_generation_boundaries,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Method display names (matching 8.0's METHOD_NAMES)
# ---------------------------------------------------------------------------
METHOD_NAMES = {
    "Vanilla": "Vanilla",
    "PredictiveEntropy": "Logit Entropy",
    "Perplexity": "Perplexity",
    "LLM-Check-attention": "LLM-Check (att.)",
    "LLM-Check-hidden": "LLM-Check (hid.)",
    "HaluNet": "HaluNet",
    "Signal-router_entropy": "Router Entropy",
    "Signal-expert_hidden_scores": "Expert Hidden",
    "Signal-expert_similarities": "Expert Similarity",
    "Signal-expert_usage_entropy": "Usage Entropy",
    "Signal-expert_usage_gini": "Usage Gini",
    "Signal-expert_usage_effective_experts": "Inv. Herfindahl",
    "Ours-LogisticRegression": "InnerExpert (LR)",
    "Ours-MLP": "InnerExpert (MLP)",
    "Ours-RandomForest": "InnerExpert (RF)",
    "Ours-XGBoost": "InnerExpert (XGBoost)",
    "Ours-Transformer": "InnerExpert (Transformer)",
    "SemanticUncertainty": "Semantic Uncertainty",
    "SemanticEnergy": "Semantic Energy",
    "SelfCheckGPT-NLI": "SelfCheckGPT (NLI)",
    "SelfCheckGPT-PROMPT": "SelfCheckGPT (Prompt)",
}

# ---------------------------------------------------------------------------
# Generation configs: each defines the output_* flags and generation mode
# ---------------------------------------------------------------------------
CONFIGS = {
    "vanilla": {
        "gen_kwargs": {"output_attentions": False, "output_hidden_states": False,
                       "output_scores": False, "output_router_logits": False},
        "use_moe_monitor": False, "sampling": False,
    },
    "scores": {
        "gen_kwargs": {"output_attentions": False, "output_hidden_states": False,
                       "output_scores": True, "output_router_logits": False},
        "use_moe_monitor": False, "sampling": False,
    },
    "attn": {
        "gen_kwargs": {"output_attentions": True, "output_hidden_states": False,
                       "output_scores": True, "output_router_logits": False},
        "use_moe_monitor": False, "sampling": False,
    },
    "hidden": {
        "gen_kwargs": {"output_attentions": False, "output_hidden_states": True,
                       "output_scores": False, "output_router_logits": False},
        "use_moe_monitor": False, "sampling": False,
    },
    "hidden_scores": {
        "gen_kwargs": {"output_attentions": False, "output_hidden_states": True,
                       "output_scores": True, "output_router_logits": False},
        "use_moe_monitor": False, "sampling": False,
    },
    "full": {
        "gen_kwargs": {},  # MoEMonitor sets all flags
        "use_moe_monitor": True, "sampling": False,
    },
    "sample_scores": {
        "gen_kwargs": {"output_attentions": False, "output_hidden_states": False,
                       "output_scores": True, "output_router_logits": False},
        "use_moe_monitor": False, "sampling": True,
    },
    "sample_text": {
        "gen_kwargs": {"output_attentions": False, "output_hidden_states": False,
                       "output_scores": False, "output_router_logits": False},
        "use_moe_monitor": False, "sampling": True,
    },
}

# Methods per config (order = table row order within each model)
CONFIG_METHODS: Dict[str, List[str]] = {
    "vanilla": ["Vanilla"],
    "scores": ["PredictiveEntropy", "Perplexity"],
    "attn": ["LLM-Check-attention"],
    "hidden": ["LLM-Check-hidden"],
    "hidden_scores": ["HaluNet"],
    "full": [
        "Signal-router_entropy",
        "Signal-expert_hidden_scores",
        "Signal-expert_similarities",
        "Signal-expert_usage_entropy",
        "Signal-expert_usage_gini",
        "Signal-expert_usage_effective_experts",
        "Ours-LogisticRegression",
        "Ours-MLP",
        "Ours-RandomForest",
        "Ours-XGBoost",
        "Ours-Transformer",
    ],
    "sample_scores": ["SemanticUncertainty", "SemanticEnergy"],
    "sample_text": ["SelfCheckGPT-NLI", "SelfCheckGPT-PROMPT"],
}

# Feature keys used by the InnerExpert detector (matching evaluate_detector)
DETECTOR_FEATURE_KEYS = [
    "hidden_scores", "attention_scores", "router_entropy",
    "expert_hidden_scores", "expert_similarities", "expert_usage",
]

# Individual signal function mapping:
# (requires_standardize_key, metric_fn, extra_args_fn or None)
INDIVIDUAL_SIGNAL_FNS = {
    "Signal-router_entropy": (
        lambda std: topk_entropy(std["expert_weights"], softmax=False),
    ),
    "Signal-expert_hidden_scores": (
        lambda std: _expert_hidden_score(
            std["expert_hidden_states"], std["expert_weights"],
        ),
    ),
    "Signal-expert_similarities": (
        lambda std: _expert_similarity_score(
            std["expert_hidden_states"], std["expert_weights"],
        ),
    ),
    "Signal-expert_usage_entropy": (
        lambda std: topk_entropy(
            _expert_usage_frequency(std["expert_idx"], weights=std["expert_weights"]),
            softmax=False,
        ),
    ),
    "Signal-expert_usage_gini": (
        lambda std: _expert_usage_gini_impurity(
            _expert_usage_frequency(std["expert_idx"], weights=std["expert_weights"]),
        ),
    ),
    "Signal-expert_usage_effective_experts": (
        lambda std: _inverse_herfindahl_index(
            _expert_usage_frequency(std["expert_idx"], weights=std["expert_weights"]),
        ),
    ),
}

# InnerExpert variant names → pkl filenames
INNEREXPERT_VARIANTS = {
    "Ours-LogisticRegression": "detector_LogisticRegression.pkl",
    "Ours-MLP": "detector_MLP.pkl",
    "Ours-RandomForest": "detector_RandomForest.pkl",
    "Ours-XGBoost": "detector_XGBoost.pkl",
    "Ours-Transformer": "detector_Transformer.pkl",
}

SELF_CHECK_PROMPT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync_and_time():
    """Synchronize CUDA and return wall-clock time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _peak_mem_gb():
    """Current peak GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    return 0.0


def _reset_mem():
    """Reset peak memory stats."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, quantize="4-bit"):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(resolve_cache_dir(model_name))
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch.cuda.empty_cache()
    quantization_kwargs = get_quantization_kwargs(quantize)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        cache_dir=str(resolve_cache_dir(model_name)),
        **quantization_kwargs,
    )
    model.eval()

    moe_monitor = MoEMonitor(
        model=model, tokenizer=tokenizer,
        output_router_logits=False, output_experts_hidden=True,
    )
    return model, tokenizer, moe_monitor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_single_shot(
    model, moe_monitor, tokenizer, input_ids_all, attention_mask_all,
    config_name, batch_size, max_new_tokens,
):
    """Run single-shot (greedy) generation for all questions.

    Returns list of per-batch dicts with 'input_ids' and 'std' (standardized
    outputs on CPU), plus gen_time and gen_peak.
    """
    config = CONFIGS[config_name]
    all_outputs = []

    _reset_mem()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(0, input_ids_all.shape[0], batch_size):
            input_ids = input_ids_all[i : i + batch_size].to(DEVICE)
            attention_mask = attention_mask_all[i : i + batch_size].to(DEVICE)

            if config["use_moe_monitor"]:
                output = moe_monitor.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            else:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **config["gen_kwargs"],
                )

            std = standardize_outputs(output, device="cpu")
            std["input_ids"] = input_ids_all[i : i + batch_size].cpu()
            all_outputs.append(std)

            del output, input_ids, attention_mask
            torch.cuda.empty_cache()

    gen_time = _sync_and_time() - t0
    gen_peak = _peak_mem_gb()

    return all_outputs, gen_time, gen_peak


def run_sampling(
    model, tokenizer, input_ids_all, attention_mask_all,
    config_name, batch_size, max_new_tokens, num_samples,
    temperature=0.7, top_p=0.9,
):
    """Run N stochastic sampling passes for all questions.

    Returns a dict with per-question decoded responses, token IDs, and
    per-token logprobs (if output_scores), plus gen_time and gen_peak.
    """
    config = CONFIGS[config_name]
    n_questions = input_ids_all.shape[0]
    gen_kwargs_sample = {
        "do_sample": True, "temperature": temperature, "top_p": top_p,
        **config["gen_kwargs"],
    }

    # Per-question lists of (sample_idx → data)
    responses_by_qid: Dict[int, List[str]] = {}
    sequences_by_qid: Dict[int, List[torch.Tensor]] = {}
    logprobs_by_qid: Dict[int, List[List[float]]] = {}  # per token
    logits_by_qid: Dict[int, List[List[float]]] = {}    # per token (raw)

    _reset_mem()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(0, n_questions, batch_size):
            input_ids = input_ids_all[i : i + batch_size].to(DEVICE)
            attention_mask = attention_mask_all[i : i + batch_size].to(DEVICE)
            bs = input_ids.shape[0]

            batch_responses = [[] for _ in range(bs)]
            batch_seq_lens = [[] for _ in range(bs)]
            batch_logprobs = [[] for _ in range(bs)]
            batch_logits = [[] for _ in range(bs)]

            for s in range(num_samples):
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    use_cache=True,
                    **gen_kwargs_sample,
                )
                std = standardize_outputs(output, device="cpu")
                std["input_ids"] = input_ids_all[i : i + batch_size].cpu()

                gen_seq_len = std["sequences"].shape[1] - std["input_ids"].shape[1]

                for b in range(bs):
                    gen_start, gen_end = find_generation_boundaries(
                        std["input_ids"][b], std["sequences"][b],
                    )
                    gen_len = gen_end - gen_start
                    batch_seq_lens[b].append(gen_len)

                    gen_ids = std["sequences"][b, gen_start:gen_end].cpu()
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    batch_responses[b].append(text)

                    if "scores" in std:
                        scores_b = std["scores"][b]  # (gen_seq_len, vocab)
                        log_probs = torch.log_softmax(scores_b, dim=-1)
                        gen_token_ids = std["sequences"][b, gen_start:gen_end].unsqueeze(-1).to(scores_b.device)
                        token_lps = torch.gather(log_probs[:gen_len], dim=-1, index=gen_token_ids[:gen_len]).squeeze(-1)
                        batch_logprobs[b].append(token_lps.cpu().tolist())
                        # Raw logits per generated token (for Semantic Energy)
                        batch_logits[b].append(scores_b[:gen_len].cpu().tolist())

                del output, std
                torch.cuda.empty_cache()

            for b in range(bs):
                qid = i + b
                responses_by_qid[qid] = batch_responses[b]
                sequences_by_qid[qid] = batch_seq_lens[b]
                if batch_logprobs[b]:
                    logprobs_by_qid[qid] = batch_logprobs[b]
                    logits_by_qid[qid] = batch_logits[b]

    gen_time = _sync_and_time() - t0
    gen_peak = _peak_mem_gb()

    return {
        "responses": responses_by_qid,
        "seq_lens": sequences_by_qid,
        "logprobs": logprobs_by_qid,
        "logits": logits_by_qid,
    }, gen_time, gen_peak


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_displayed_tokens_single(all_outputs):
    """Count displayed (= generated) tokens across all batches."""
    total = 0
    for std in all_outputs:
        input_ids = std["input_ids"]
        sequences = std["sequences"]
        for b in range(input_ids.shape[0]):
            gen_start, gen_end = find_generation_boundaries(input_ids[b], sequences[b])
            total += gen_end - gen_start
    return total


def count_displayed_tokens_sampled(sampled_data):
    """Count displayed tokens from sample 0 only."""
    total = 0
    for qid, gen_lens in sampled_data["seq_lens"].items():
        if gen_lens:
            total += gen_lens[0]
    return total


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _ensure_standardized(std):
    """Ensure std has input_ids on CPU (already should from run_single_shot)."""
    return std


def score_predictive_entropy(all_outputs):
    """Time topk_entropy on output scores."""
    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()
    for std in all_outputs:
        if "scores" not in std:
            continue
        _ = topk_entropy(std["scores"].to(DEVICE))
    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    return score_time, score_peak


def score_perplexity(all_outputs):
    """Time compute_baseline_features (perplexity extraction)."""
    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()
    for std in all_outputs:
        if "scores" not in std:
            continue
        _ = compute_baseline_features({k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in std.items()})
    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    return score_time, score_peak


def score_llm_check_attention(all_outputs):
    """Time attention_score on raw attentions."""
    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()
    for std in all_outputs:
        if "attentions" not in std:
            continue
        _ = _attention_score(std["attentions"].to(DEVICE))
    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    return score_time, score_peak


def score_llm_check_hidden(all_outputs):
    """Time hidden_score (SVD) on hidden states."""
    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()
    for std in all_outputs:
        if "hidden_states" not in std:
            continue
        _ = _hidden_score(std["hidden_states"].to(DEVICE))
    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    return score_time, score_peak


def score_halunet(all_outputs, models_dir, model_slug):
    """Time HaluNet: compute_baseline_features + last_hidden_states + predict_proba."""
    from moeuncert.baselines import HaluNet

    halunet = HaluNet(device=DEVICE)
    halunet_path = Path(models_dir) / model_slug / "halunet.pt"
    halunet.load(halunet_path)

    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()

    total_tokens = 0
    for std in all_outputs:
        if "scores" not in std or "hidden_states" not in std:
            continue
        std_dev = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in std.items()}
        features = compute_metrics(std_dev, return_baseline_features=True)

        input_ids = std["input_ids"]
        sequences = std["sequences"]
        B = input_ids.shape[0]
        for b in range(B):
            gen_start, gen_end = find_generation_boundaries(input_ids[b], sequences[b])
            gen_len = gen_end - gen_start
            if gen_len <= 0:
                continue
            ll = features["log_likelihoods"][b, :gen_len].cpu().numpy()
            ent = features["entropies"][b, :gen_len].cpu().numpy()
            hs = features["last_hidden_states"][b, :gen_len, :].cpu().numpy()
            _ = halunet.predict_proba(ll, ent, hs)

    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    del halunet
    torch.cuda.empty_cache()
    return score_time, score_peak


def score_individual_signal(all_outputs, signal_name):
    """Time a single individual MoE signal: standardize + compute that signal."""
    signal_fn = INDIVIDUAL_SIGNAL_FNS[signal_name]

    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()

    for std in all_outputs:
        if "expert_weights" not in std:
            continue
        std_dev = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in std.items()}
        _ = signal_fn(std_dev)

    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    return score_time, score_peak


def _build_detector_X_tokens(std, metrics, feature_names):
    """Build X_tokens for the detector from standardized outputs + metrics.

    Replicates the logic from evaluate_detector in predictions.py.
    """
    input_ids = std["input_ids"]
    sequences = std["sequences"]
    B = input_ids.shape[0]

    all_X = []
    for b in range(B):
        gen_start, gen_end = find_generation_boundaries(input_ids[b], sequences[b])
        gen_len = gen_end - gen_start
        if gen_len <= 0:
            continue

        token_feats = {}
        for key in DETECTOR_FEATURE_KEYS:
            if key not in metrics:
                continue
            tensor = metrics[key][b]
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
            elif key in ("expert_similarities", "expert_hidden_scores"):
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
            token_feat_arrays[key] = arr

        X_tokens = np.zeros((gen_len, len(feature_names)), dtype=np.float32)
        for col_idx, fname in enumerate(feature_names):
            matched_key = next(
                (k for k in DETECTOR_FEATURE_KEYS if fname.startswith(f"{k}_")), None,
            )
            if matched_key is None:
                continue
            suffix = fname[len(matched_key) + 1:]
            if not suffix or not suffix.isdigit():
                continue
            feat_idx = int(suffix)
            arr = token_feat_arrays.get(matched_key)
            if arr is not None and feat_idx < arr.shape[1]:
                X_tokens[:, col_idx] = arr[:, feat_idx]

        all_X.append(X_tokens)

    return all_X


def score_inner_expert(all_outputs, models_dir, model_slug, variant_name, pkl_filename):
    """Time InnerExpert variant: standardize + compute_metrics + build X + predict_proba."""
    detector_path = Path(models_dir) / model_slug / pkl_filename
    with open(detector_path, "rb") as f:
        detector_data = pickle.load(f)
    model = detector_data["model"]
    feature_names = detector_data["feature_names"]

    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()

    for std in all_outputs:
        if "expert_weights" not in std:
            continue
        std_dev = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in std.items()}
        metrics = compute_metrics(std_dev, return_baseline_features=False)
        all_X = _build_detector_X_tokens(std, metrics, feature_names)
        for X_tokens in all_X:
            X_safe = np.nan_to_num(X_tokens, nan=0.0, posinf=1e10, neginf=-1e10)
            _ = model.predict_proba(X_safe)

    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    del model
    return score_time, score_peak


def score_semantic_uncertainty(sampled_data, tokenizer):
    """Time Semantic Uncertainty: NLI clustering + logsumexp computation."""
    from moeuncert.baselines import SemanticUncertainty

    su = SemanticUncertainty()
    responses_by_qid = sampled_data["responses"]
    logprobs_by_qid = sampled_data["logprobs"]

    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()

    for qid in sorted(responses_by_qid.keys()):
        responses = responses_by_qid[qid]
        lps = logprobs_by_qid.get(qid, [])
        if len(responses) < 2 or not lps:
            continue
        try:
            _ = su.predict_proba(responses, lps)
        except Exception:
            pass

    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    del su
    torch.cuda.empty_cache()
    return score_time, score_peak


def score_semantic_energy(sampled_data, tokenizer):
    """Time Semantic Energy: NLI clustering + energy computation."""
    from moeuncert.baselines import SemanticEnergy, SemanticUncertainty
    from moeuncert.baselines.semantic_uncertainty import semantic_ids_to_clusters

    su = SemanticUncertainty()
    se = SemanticEnergy()
    responses_by_qid = sampled_data["responses"]
    logprobs_by_qid = sampled_data["logprobs"]
    logits_by_qid = sampled_data["logits"]

    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()

    for qid in sorted(responses_by_qid.keys()):
        responses = responses_by_qid[qid]
        lps = logprobs_by_qid.get(qid, [])
        raw_logits = logits_by_qid.get(qid, [])
        if len(responses) < 2 or not raw_logits or not lps:
            continue
        try:
            semantic_ids = su.cluster_responses(responses)
            clusters = semantic_ids_to_clusters(semantic_ids)
            response_probs = [[np.exp(lp) for lp in ll] for ll in lps]
            _ = se.predict_proba(
                response_logits=raw_logits,
                response_probs=response_probs,
                clusters=clusters,
            )
        except Exception:
            pass

    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    del su, se
    torch.cuda.empty_cache()
    return score_time, score_peak


def score_selfcheck_nli(sampled_data, tokenizer):
    """Time SelfCheckGPT NLI: DeBERTa NLI inference per (sentence, sample) pair."""
    from moeuncert.baselines import SelfCheckNLI

    checker = SelfCheckNLI(device=DEVICE)
    responses_by_qid = sampled_data["responses"]

    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()

    for qid in sorted(responses_by_qid.keys()):
        responses = responses_by_qid[qid]
        if len(responses) < 2:
            continue
        target = responses[0]
        sampled_passages = responses[1:]
        try:
            _ = checker.predict_proba([target], sampled_passages)
        except Exception:
            pass

    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    del checker
    torch.cuda.empty_cache()
    return score_time, score_peak


def score_selfcheck_prompt(sampled_data, tokenizer, model_name):
    """Time SelfCheckGPT Prompt: LLM scoring per (sentence, sample) pair.

    Uses Llama-2-7b-chat-hf as the scoring LLM (we have HF access).
    """
    from moeuncert.baselines import SelfCheckPrompt

    checker = SelfCheckPrompt(model=SELF_CHECK_PROMPT_MODEL, device=DEVICE)
    responses_by_qid = sampled_data["responses"]

    _sync_and_time()
    _reset_mem()
    t0 = time.perf_counter()

    for qid in sorted(responses_by_qid.keys()):
        responses = responses_by_qid[qid]
        if len(responses) < 2:
            continue
        target = responses[0]
        sampled_passages = responses[1:]
        try:
            _ = checker.predict_proba([target], sampled_passages)
        except Exception:
            pass

    score_time = _sync_and_time() - t0
    score_peak = _peak_mem_gb()
    del checker
    torch.cuda.empty_cache()
    return score_time, score_peak


# ---------------------------------------------------------------------------
# Scoring dispatcher
# ---------------------------------------------------------------------------

def time_method(
    method_name, config_name, all_outputs, sampled_data,
    tokenizer, models_dir, model_slug, model_name,
):
    """Dispatch to the correct scoring function and return (score_time, score_peak)."""
    if method_name == "Vanilla":
        return 0.0, 0.0

    if method_name == "PredictiveEntropy":
        return score_predictive_entropy(all_outputs)
    if method_name == "Perplexity":
        return score_perplexity(all_outputs)
    if method_name == "LLM-Check-attention":
        return score_llm_check_attention(all_outputs)
    if method_name == "LLM-Check-hidden":
        return score_llm_check_hidden(all_outputs)
    if method_name == "HaluNet":
        return score_halunet(all_outputs, models_dir, model_slug)

    if method_name in INDIVIDUAL_SIGNAL_FNS:
        return score_individual_signal(all_outputs, method_name)

    if method_name in INNEREXPERT_VARIANTS:
        return score_inner_expert(
            all_outputs, models_dir, model_slug,
            method_name, INNEREXPERT_VARIANTS[method_name],
        )

    if method_name == "SemanticUncertainty":
        return score_semantic_uncertainty(sampled_data, tokenizer)
    if method_name == "SemanticEnergy":
        return score_semantic_energy(sampled_data, tokenizer)
    if method_name == "SelfCheckGPT-NLI":
        return score_selfcheck_nli(sampled_data, tokenizer)
    if method_name == "SelfCheckGPT-PROMPT":
        return score_selfcheck_prompt(sampled_data, tokenizer, model_name)

    raise ValueError(f"Unknown method: {method_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="7.2 — Inference time & memory benchmark"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "allenai/OLMoE-1B-7B-0924-Instruct",
            "google/gemma-4-26B-A4B-it",
        ],
        help="HuggingFace model names",
    )
    parser.add_argument("--quantize", default="4-bit", choices=["16-bit", "8-bit", "4-bit"])
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/inference-benchmark"))
    parser.add_argument("--rt-year", type=int, default=2026)
    parser.add_argument("--rt-month", type=int, default=6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("7.2 — INFERENCE TIME & MEMORY BENCHMARK")
    print("=" * 70)
    print(f"Models:      {args.models}")
    print(f"Quantize:    {args.quantize}")
    print(f"Questions:   {args.num_questions} (RealtimeQA {args.rt_year}/{args.rt_month:02d})")
    print(f"Max tokens:  {args.max_new_tokens}")
    print(f"Batch size:  {args.batch_size} (single-shot), {args.sample_batch_size} (sampling)")
    print(f"Num samples: {args.num_samples}")
    print(f"GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- Fetch dataset (shared across all models) ---
    print("\n[1/3] Fetching RealtimeQA data...")
    df = fetch_realtimeqa(split=args.rt_year, month=args.rt_month)
    df = df.sample(n=min(args.num_questions, len(df)), random_state=42).reset_index(drop=True)
    print(f"  {len(df)} questions")

    for model_name in args.models:
        model_slug = resolve_model_slug(model_name)
        print(f"\n{'=' * 70}")
        print(f"[2/3] Benchmarking {model_name} ({model_slug})")
        print(f"{'=' * 70}")

        model, tokenizer, moe_monitor = load_model_and_tokenizer(model_name, args.quantize)

        # Tokenize
        inputs = tokenize_realtimeqa(tokenizer, df, with_evidence=False)
        input_ids_all = inputs["input_ids"]
        attention_mask_all = inputs["attention_mask"]

        results = []

        # --- Single-shot configs ---
        single_shot_order = ["vanilla", "scores", "attn", "hidden", "hidden_scores", "full"]
        for config_name in single_shot_order:
            print(f"\n  Config: {config_name}")
            all_outputs, gen_time, gen_peak = run_single_shot(
                model, moe_monitor, tokenizer,
                input_ids_all, attention_mask_all,
                config_name, args.batch_size, args.max_new_tokens,
            )
            total_tokens = count_displayed_tokens_single(all_outputs)
            print(f"    gen_time={gen_time:.2f}s, tokens={total_tokens}, gen_peak={gen_peak:.2f}GB")

            for method_name in CONFIG_METHODS[config_name]:
                print(f"    Method: {method_name}...", end=" ", flush=True)
                try:
                    score_time, score_peak = time_method(
                        method_name, config_name, all_outputs, None,
                        tokenizer, args.models_dir, model_slug, model_name,
                    )
                except Exception as e:
                    print(f"FAILED ({e})")
                    score_time = float("nan")
                    score_peak = float("nan")

                peak_mem = max(gen_peak, score_peak) if not np.isnan(score_peak) else gen_peak
                total_time = gen_time + (score_time if not np.isnan(score_time) else 0.0)
                time_per_100 = total_time * 100 / total_tokens if total_tokens > 0 else float("nan")

                row = {
                    "method": METHOD_NAMES.get(method_name, method_name),
                    "raw_method": method_name,
                    "config": config_name,
                    "num_samples": 0,
                    "num_questions": len(df),
                    "total_tokens_displayed": total_tokens,
                    "gen_time_s": round(gen_time, 4),
                    "score_time_s": round(score_time, 4) if not np.isnan(score_time) else float("nan"),
                    "total_time_s": round(total_time, 4),
                    "time_per_100_tokens_s": round(time_per_100, 6),
                    "peak_gpu_mem_gb": round(peak_mem, 2),
                }
                results.append(row)
                print(f"score_time={score_time:.3f}s, total={total_time:.2f}s, "
                      f"per100tok={time_per_100:.4f}s, peak={peak_mem:.2f}GB")

            del all_outputs
            gc.collect()
            torch.cuda.empty_cache()

        # --- Sampling configs ---
        sampling_order = ["sample_scores", "sample_text"]
        for config_name in sampling_order:
            print(f"\n  Config: {config_name}")
            sampled_data, gen_time, gen_peak = run_sampling(
                model, tokenizer,
                input_ids_all, attention_mask_all,
                config_name,
                args.sample_batch_size, args.max_new_tokens, args.num_samples,
                args.temperature, args.top_p,
            )
            total_tokens = count_displayed_tokens_sampled(sampled_data)
            print(f"    gen_time={gen_time:.2f}s, displayed_tokens={total_tokens}, gen_peak={gen_peak:.2f}GB")

            for method_name in CONFIG_METHODS[config_name]:
                print(f"    Method: {method_name}...", end=" ", flush=True)
                try:
                    score_time, score_peak = time_method(
                        method_name, config_name, None, sampled_data,
                        tokenizer, args.models_dir, model_slug, model_name,
                    )
                except Exception as e:
                    print(f"FAILED ({e})")
                    score_time = float("nan")
                    score_peak = float("nan")

                peak_mem = max(gen_peak, score_peak) if not np.isnan(score_peak) else gen_peak
                total_time = gen_time + (score_time if not np.isnan(score_time) else 0.0)
                time_per_100 = total_time * 100 / total_tokens if total_tokens > 0 else float("nan")

                row = {
                    "method": METHOD_NAMES.get(method_name, method_name),
                    "raw_method": method_name,
                    "config": config_name,
                    "num_samples": args.num_samples,
                    "num_questions": len(df),
                    "total_tokens_displayed": total_tokens,
                    "gen_time_s": round(gen_time, 4),
                    "score_time_s": round(score_time, 4) if not np.isnan(score_time) else float("nan"),
                    "total_time_s": round(total_time, 4),
                    "time_per_100_tokens_s": round(time_per_100, 6),
                    "peak_gpu_mem_gb": round(peak_mem, 2),
                }
                results.append(row)
                print(f"score_time={score_time:.3f}s, total={total_time:.2f}s, "
                      f"per100tok={time_per_100:.4f}s, peak={peak_mem:.2f}GB")

            del sampled_data
            gc.collect()
            torch.cuda.empty_cache()

        # --- Save results ---
        out_dir = args.output_dir / model_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        df_results = pd.DataFrame(results)
        csv_path = out_dir / "benchmark.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")
        print(df_results[["method", "total_time_s", "time_per_100_tokens_s", "peak_gpu_mem_gb"]].to_string(index=False))

        # --- Cleanup model ---
        del model, moe_monitor, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("7.2 — BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved to: {args.output_dir}/")
    print("Next: re-run 8.0-results-analysis.py to generate the LaTeX table.")


if __name__ == "__main__":
    main()