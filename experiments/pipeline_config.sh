#!/usr/bin/env bash
# ============================================================================
# pipeline_config.sh — shared defaults for train_pipeline.sh and eval_pipeline.sh
# ============================================================================
# This file is sourced by both pipeline scripts.  Edit these variables
# if you want to change the model, sampling parameters, or the data split.
# ============================================================================

# --- MODEL ------------------------------------------------------------------
export MODEL="allenai/OLMoE-1B-7B-0924-Instruct"
export QUANTIZE="4-bit"
export LABEL_MODEL="zai-org/GLM-5.1"

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
# ----------------------------------------------------------------------------

# Results: results.parquet
_pipeline_has_results() {
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
    [[ -f ${data_dir}/results.parquet ]]
}

# Samples: sampled_generation/*.pt
_pipeline_has_samples() {
    local year="$1"
    local month="$2"
    local model="$3"
    local slug
    slug=$(derive_model_slug "${model}")
    local sampled_dir
    if [[ -n ${month} ]]; then
        sampled_dir="data/realtimeqa-${year}-$(printf '%02d' "${month}")/${slug}/sampled_generation"
    else
        sampled_dir="data/realtimeqa-${year}/${slug}/sampled_generation"
    fi
    [[ -d ${sampled_dir} ]] && \
        [[ $(find "${sampled_dir}" -maxdepth 1 -name '*.pt' -print | wc -l) -gt 0 ]]
}

# Labels: results_labeled_*.parquet
_pipeline_has_labels() {
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
    find "${data_dir}" -maxdepth 1 -name 'results_labeled_*.parquet' | grep -q .
}

# Predictions: predictions/ directory
_pipeline_has_predictions() {
    local year="$1"
    local month="$2"
    local model="$3"
    local slug
    slug=$(derive_model_slug "${model}")
    local src
    if [[ -n ${month} ]]; then
        src="data/realtimeqa-${year}-$(printf '%02d' "${month}")/${slug}/predictions"
    else
        src="data/realtimeqa-${year}/${slug}/predictions"
    fi
    [[ -d ${src} ]]
}

# Analysis artefacts: comparison_table.md, results.json, or *.png
_pipeline_has_analysis() {
    local year="$1"
    local month="$2"
    local model="$3"
    local slug
    slug=$(derive_model_slug "${model}")
    local analysis_dir
    if [[ -n ${month} ]]; then
        analysis_dir="data/realtimeqa-${year}-$(printf '%02d' "${month}")/${slug}/analysis"
    else
        analysis_dir="data/realtimeqa-${year}/${slug}/analysis"
    fi
    [[ -d ${analysis_dir} ]] && \
        [[ $(find "${analysis_dir}" -maxdepth 1 \
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
