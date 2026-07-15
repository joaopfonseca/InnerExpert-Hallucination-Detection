#!/usr/bin/env bash
# ============================================================================
# pipeline_config.sh — shared defaults for train_pipeline.sh and eval_pipeline.sh
# ============================================================================
# This file is sourced by both pipeline scripts.  Edit these variables
# if you want to change the model, sampling parameters, or the data split.
# ============================================================================
#
# ----------------------------------------------------------------------------
# SUPPORTED MODELS
# ----------------------------------------------------------------------------
# The pipeline is driven by the HF model id in $MODEL below.  Both models
# below are registered in moeuncert.forwards.MOE_FORWARD_REGISTRY, so the
# MoE-instrumentation (router logits, expert hidden states, etc.) works for
# either one out of the box.
#
#   Model                                  HF id                                          MoE layout              Notes
#   ─────────────────────────────────────  ─────────────────────────────────────────────  ──────────────────────  ─────────────────────────────
#   OLMoE-1B-7B-0924-Instruct              allenai/OLMoE-1B-7B-0924-Instruct              64 experts, 8 active   Default. 4-bit quantization.
#   Gemma 4 26B A4B IT (text-only)         google/gemma-4-26B-A4B-it                      128+1 shared, 8 active Multimodal; we use it as CausalLM.
#
# To switch models, change $MODEL and re-source (or just re-run the pipeline).
# Quantization is set globally via $QUANTIZE; per-model recommendations are
# listed above.  Each model gets its own data and models directory derived
# from the HF id slug (so the two models do not collide on disk).
# ----------------------------------------------------------------------------

# --- MODEL ------------------------------------------------------------------
export MODEL="allenai/OLMoE-1B-7B-0924-Instruct"
export QUANTIZE="4-bit"
export LABEL_MODEL="zai-org/GLM-5.1"

# --- MODEL CACHE (HF_HOME) ---------------------------------------------------
# By default all HuggingFace downloads (models, tokenizers, datasets) are
# cached inside the project at ``pretrained_models/`` instead of the global
# ``~/.cache/huggingface/``.  This makes the repo self-contained and avoids
# filling up the user's home directory.
#
# If you prefer to use the global cache, comment out the line below or set
# HF_HOME to another directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export HF_HOME="${PROJECT_ROOT}/pretrained_models"
export TRANSFORMERS_CACHE="${HF_HOME}"

# Pass through Hugging Face auth token from default cache.
# HF_HOME redirects the token lookup path away from ~/.cache/huggingface/token.
# Re-export the token so gated repos (e.g. Llama-2 for SelfCheckPrompt) work.
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

# --- GENERATION --------------------------------------------------------------
export MAX_NEW_TOKENS=65

# --- SAMPLING ----------------------------------------------------------------
export NUM_SAMPLES=5
export SAMPLING_TEMPERATURE=0.7
export SAMPLING_TOP_P=0.9
export SAMPLING_BATCH_SIZE=2

# --- LABELING ----------------------------------------------------------------
export ANSWER_THRESHOLD=0.5

# --- BASELINE FEATURES (required for HaluNet + perplexity computations) ------
export RETURN_BASELINE_FEATURES="--return-baseline-features"

# --- DATA SPLIT ---------------------------------------------------------------
export TRAIN_YEARS=(2024 2025)
export TRAIN_MONTH=""      # empty = all months; e.g. 1 for January only
export TEST_YEARS=(2026)
export TEST_MONTH=""       # empty = all months; e.g. 1 for January only

# --- HELPER: derive MODEL_SLUG from MODEL_NAME -------------------------------
# Usage: export MODEL_SLUG=$(derive_model_slug "${MODEL}")
derive_model_slug() {
    local name="$1"
    echo "${name//\//__}"
}

# ----------------------------------------------------------------------------
# DATA SKIP-CHECK HELPERS (used by both train_pipeline.sh and eval_pipeline.sh)
# Each function inspects the data directory for a given YEAR / MONTH / MODEL.
# All per-year helpers also accept a "combined" dir (e.g.
# ``realtimeqa-2024-2025/{slug}``) that covers the requested year, mirroring
# the loader's fallback in ``load_multi_year_data``.
# ----------------------------------------------------------------------------

# Find the per-model subdir of a combined dataset dir on disk that *covers*
# the requested year.  MONTH may be empty.  Echoes the absolute path, or
# nothing if no such combined dir exists.
#
# Mirrors the parsing and discovery logic in
# ``moeuncert.experiments.data_loading._find_combined_dataset_dir`` (so the
# skip-check matches what the loader would actually find at read time), but
# intentionally does NOT call ``_has_data_files`` — the loader uses that
# helper to require *source* data (results/labels/base_generation), while the
# pipeline skip-check should also accept downstream artefacts (predictions,
# analysis) that may have been written by an earlier phase.
#
# Usage: _pipeline_combined_dir_for_year YEAR MONTH MODEL
_pipeline_combined_dir_for_year() {
    local year="$1"
    local month="$2"
    local model="$3"
    python - "$year" "$month" "$model" <<'PY'
import re, sys
from pathlib import Path

year_s, month_s, model = sys.argv[1], sys.argv[2], sys.argv[3]
month = int(month_s) if month_s else None
data_root = Path("data")
if not data_root.is_dir():
    sys.exit(0)
slug = model.replace("/", "__")

# Mirror moeuncert.experiments.data_loading._parse_dataset_dir_name.
def parse(name):
    m = re.fullmatch(r"realtimeqa-(\d{4}(?:-\d{4})*)(?:-(\d{2}))?", name)
    if not m:
        return None
    years = [int(y) for y in m.group(1).split("-")]
    candidate_month = int(m.group(2)) if m.group(2) else None
    if candidate_month is not None and len(years) != 1:
        return None
    return years, candidate_month

# Mirror _find_combined_dataset_dir's filtering and "smallest superset" tie-break.
candidates = []
for d in data_root.iterdir():
    if not d.is_dir():
        continue
    parsed = parse(d.name)
    if parsed is None:
        continue
    c_years, c_month = parsed
    if month is not None:
        exact = c_month == month and int(year_s) in c_years
        fallback = c_month is None and int(year_s) in c_years
        if not (exact or fallback):
            continue
    else:
        if c_month is not None:
            continue
        if int(year_s) not in c_years:
            continue
    model_dir = d / slug
    if model_dir.is_dir():
        candidates.append((len(c_years), model_dir))

if not candidates:
    sys.exit(0)
candidates.sort()
print(candidates[0][1])
PY
}

# Echoes the combined-dir path (if any) for YEAR/MONTH/MODEL, or "" if none.
# Centralises the fallback so per-year helpers stay short.
_pipeline_resolve_data_dir() {
    local year="$1"
    local month="$2"
    local model="$3"
    local slug
    slug=$(derive_model_slug "${model}")
    local data_dir
    if [[ -n ${month} ]]; then
        data_dir="data/realtimeqa-${year}-$(printf '%02d' "${month}")/${slug}"
    else
        data_dir="data/realtimeqa-${year}/${slug}"
    fi
    if [[ -d ${data_dir} ]]; then
        echo "${data_dir}"
        return 0
    fi
    # Fall back to a combined dir (e.g. realtimeqa-2024-2025/) that covers year.
    local combined
    if combined=$(_pipeline_combined_dir_for_year "${year}" "${month}" "${model}"); then
        echo "${combined}"
    fi
}

# Results: results.parquet
_pipeline_has_results() {
    local year="$1"
    local month="$2"
    local model="$3"
    local data_dir
    data_dir=$(_pipeline_resolve_data_dir "${year}" "${month}" "${model}")
    [[ -n ${data_dir} && -f ${data_dir}/results.parquet ]]
}

# Samples: sampled_generation/*.pt
_pipeline_has_samples() {
    local year="$1"
    local month="$2"
    local model="$3"
    local data_dir
    data_dir=$(_pipeline_resolve_data_dir "${year}" "${month}" "${model}")
    [[ -n ${data_dir} && -d ${data_dir}/sampled_generation ]] && \
        [[ $(find "${data_dir}/sampled_generation" -maxdepth 1 -name '*.pt' -print | wc -l) -gt 0 ]]
}

# Labels: results_labeled_*.parquet
_pipeline_has_labels() {
    local year="$1"
    local month="$2"
    local model="$3"
    local data_dir
    data_dir=$(_pipeline_resolve_data_dir "${year}" "${month}" "${model}")
    [[ -n ${data_dir} ]] && \
        find "${data_dir}" -maxdepth 1 -name 'results_labeled_*.parquet' | grep -q .
}

# Predictions: predictions/ directory
_pipeline_has_predictions() {
    local year="$1"
    local month="$2"
    local model="$3"
    local data_dir
    data_dir=$(_pipeline_resolve_data_dir "${year}" "${month}" "${model}")
    [[ -n ${data_dir} && -d ${data_dir}/predictions ]]
}

# Analysis artefacts: comparison_table.md, results.json, or *.png
_pipeline_has_analysis() {
    local year="$1"
    local month="$2"
    local model="$3"
    local data_dir
    data_dir=$(_pipeline_resolve_data_dir "${year}" "${month}" "${model}")
    [[ -n ${data_dir} && -d ${data_dir}/analysis ]] && \
        [[ $(find "${data_dir}/analysis" -maxdepth 1 \
            \( -name 'comparison_table.md' -o -name 'results.json' -o -name '*.png' \) \
            -print | wc -l) -gt 0 ]]
}

# Aggregate checks: all years must pass the per-year check
all_years_have_results() {
    local month="$1"
    shift
    for year in "$@"; do
        _pipeline_has_results "${year}" "${month}" "${MODEL}" || return 1
    done
    return 0
}

all_years_have_samples() {
    local month="$1"
    shift
    for year in "$@"; do
        _pipeline_has_samples "${year}" "${month}" "${MODEL}" || return 1
    done
    return 0
}

all_years_have_labels() {
    local month="$1"
    shift
    for year in "$@"; do
        _pipeline_has_labels "${year}" "${month}" "${MODEL}" || return 1
    done
    return 0
}

all_years_have_predictions() {
    local month="$1"
    shift
    for year in "$@"; do
        _pipeline_has_predictions "${year}" "${month}" "${MODEL}" || return 1
    done
    return 0
}

all_years_have_analysis() {
    local month="$1"
    shift
    for year in "$@"; do
        _pipeline_has_analysis "${year}" "${month}" "${MODEL}" || return 1
    done
    return 0
}
