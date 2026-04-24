#!/usr/bin/env bash
# ============================================================================
# End-to-end pipeline runner for MoE Uncertainty Estimation experiments
#
# This script defines all configurable variables at the top, then runs each
# experiment script in sequence with the appropriate flags.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# You can override any variable by setting it before running the script:
#   MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" ./run_pipeline.sh
# ============================================================================

set -euo pipefail

# --- DATASET CONFIGURATION ---
YEARS=(2025 2026)
MONTH=""                        # Leave empty to use previous month
DATASET="realtimeqa"

# --- MODEL CONFIGURATION ---
MODEL="allenai/OLMoE-1B-7B-0924-Instruct"
QUANTIZE="4-bit"                # Options: "16-bit", "8-bit", "4-bit"

# --- GENERATION CONFIGURATION ---
MAX_NEW_TOKENS=65
BATCH_SIZE=6

# --- SAMPLING CONFIGURATION (for sampling-based baselines) ---
NUM_SAMPLES=5                   # Number of stochastically sampled responses per question
SAMPLING_TEMPERATURE=0.7
SAMPLING_TOP_P=0.9
SAMPLING_BATCH_SIZE=2

# --- LABELING CONFIGURATION ---
ANSWER_THRESHOLD=0.5            # Threshold for binary hallucination labels
LLM_MODEL="zai-org/GLM-5.1"   # LLM-as-a-Judge model
USE_LLM_TOKEN_LABELS=true       # Set to true for token-level LLM labels
MAX_LLM_ANSWER_SAMPLES=""       # Limit LLM answer queries (empty = all)
MAX_LLM_TOKEN_SAMPLES=""        # Limit LLM token queries (empty = all)

# --- COMPUTATION FLAGS ---
RETURN_BASELINE_FEATURES=true   # Set to true to compute log-likelihoods + entropies for HaluNet

# --- PATHS ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="${SCRIPT_DIR}/experiments"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# ============================================================================
# Helper functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_script() {
    local script_name="$1"
    shift
    log "Running: ${script_name} $*"
    "${VENV_PYTHON}" "${EXPERIMENTS_DIR}/${script_name}" "$@"
}

# ============================================================================
# STEP 1: Generate answers (single-pass, greedy generation)
# ============================================================================

log "=== Step 1: Generate answers (base generation) ==="

run_script "1.0-generate-answers.py" \
    --model "${MODEL}" \
    --quantize "${QUANTIZE}" \
    --years "${YEARS[@]}" \
    $( [ -n "${MONTH}" ] && echo "--month ${MONTH}" ) \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}"

# ============================================================================
# STEP 2: Generate sampled responses (for sampling-based baselines)
# ============================================================================

log "=== Step 2: Generate sampled responses for SU/SE/SCGPT ==="

run_script "1.1-generate-baseline-samples.py" \
    --model "${MODEL}" \
    --quantize "${QUANTIZE}" \
    --years "${YEARS[@]}" \
    $( [ -n "${MONTH}" ] && echo "--month ${MONTH}" ) \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-samples "${NUM_SAMPLES}" \
    --temperature "${SAMPLING_TEMPERATURE}" \
    --top-p "${SAMPLING_TOP_P}" \
    --batch-size "${SAMPLING_BATCH_SIZE}"

# ============================================================================
# STEP 3: Generate labels (ROUGE/BLEU-based + optional LLM-as-Judge)
# ============================================================================

log "=== Step 3: Generate hallucination labels ==="

LLM_ARGS=()
if [ "${USE_LLM_TOKEN_LABELS}" = true ]; then
    LLM_ARGS+=(--use-llm-token-labels)
fi
if [ -n "${MAX_LLM_ANSWER_SAMPLES}" ]; then
    LLM_ARGS+=(--max-llm-answer-samples "${MAX_LLM_ANSWER_SAMPLES}")
fi
if [ -n "${MAX_LLM_TOKEN_SAMPLES}" ]; then
    LLM_ARGS+=(--max-llm-token-samples "${MAX_LLM_TOKEN_SAMPLES}")
fi

run_script "2.0-make-labels.py" \
    --model "${MODEL}" \
    --years "${YEARS[@]}" \
    $( [ -n "${MONTH}" ] && echo "--month ${MONTH}" ) \
    --answer-threshold "${ANSWER_THRESHOLD}" \
    --deepinfra-model "${LLM_MODEL}" \
    "${LLM_ARGS[@]}"

# ============================================================================
# STEP 4: Analyze labels and metrics
# ============================================================================

log "=== Step 4: Analyze labels and metrics ==="

run_script "2.1-analyze-labels.py" \
    --model "${MODEL}" \
    --years "${YEARS[@]}" \
    $( [ -n "${MONTH}" ] && echo "--month ${MONTH}" )

run_script "1.1-analyze-metrics.py" \
    --model "${MODEL}" \
    --years "${YEARS[@]}" \
    $( [ -n "${MONTH}" ] && echo "--month ${MONTH}" )

# ============================================================================
# STEP 5: Train detection model (your MoE method)
# ============================================================================

log "=== Step 5: Train MoE detection model ==="

run_script "3.0-detection-model-training.py" \
    --model "${MODEL}" \
    --years "${YEARS[@]}" \
    $( [ -n "${MONTH}" ] && echo "--month ${MONTH}" ) \
    --answer-threshold "${ANSWER_THRESHOLD}"

log "=== Pipeline complete! ==="
log "Results saved in data/<dataset>/<model_slug>/"
log "Sampling outputs saved in data/<dataset>/<model_slug>/sampled_generation/"
