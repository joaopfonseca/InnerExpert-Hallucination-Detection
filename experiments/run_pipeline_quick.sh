#!/usr/bin/env bash
# ============================================================================
# Quick pipeline: train on 2025, test OOD on January 2026.
#
# A faster variant of run_pipeline.sh for rapid iteration.
# Uses a single training year instead of 2022-2025.
#
# Usage:
#   chmod +x run_pipeline_quick.sh
#   ./run_pipeline_quick.sh
# ============================================================================

set -euo pipefail

# --- GENERATION (both years needed — train + test) ---
GENERATION_YEARS=(2025 2026)
GENERATION_MONTH=""             # Empty = all months for each year

# --- TRAINING (single year) ---
TRAIN_YEARS=(2025)
TRAIN_MONTH=""                  # Empty = all months

# --- TEST (single month) ---
TEST_YEARS=(2026)
TEST_MONTH=1                    # January 2026

# --- MODEL CONFIGURATION ---
MODEL="allenai/OLMoE-1B-7B-0924-Instruct"
QUANTIZE="4-bit"
LABEL_MODEL="zai-org/GLM-5.1"

# --- GENERATION ---
MAX_NEW_TOKENS=65

# --- SAMPLING ---
NUM_SAMPLES=5
SAMPLING_TEMPERATURE=0.7
SAMPLING_TOP_P=0.9
SAMPLING_BATCH_SIZE=2

# --- LABELING ---
ANSWER_THRESHOLD=0.5

# --- BASELINE FEATURES (required for HaluNet + perplexity) ---
RETURN_BASELINE_FEATURES="--return-baseline-features"

# --- PATHS ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Helpers
# ============================================================================

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }

run_script() {
    local script="$1"; shift
    log "Running: ${script} $*"
    python "${SCRIPT_DIR}/${script}" "$@"
}

# ============================================================================
# PHASE 1: GENERATE
# ============================================================================

log "=== PHASE 1: Generate answers ==="

run_script "1.0-generate-answers.py" \
    --model "${MODEL}" \
    --quantize "${QUANTIZE}" \
    --years "${GENERATION_YEARS[@]}" \
    $( [ -n "${GENERATION_MONTH}" ] && echo "--month ${GENERATION_MONTH}" ) \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    ${RETURN_BASELINE_FEATURES}

run_script "1.1-generate-baseline-samples.py" \
    --model "${MODEL}" \
    --quantize "${QUANTIZE}" \
    --years "${GENERATION_YEARS[@]}" \
    $( [ -n "${GENERATION_MONTH}" ] && echo "--month ${GENERATION_MONTH}" ) \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-samples "${NUM_SAMPLES}" \
    --temperature "${SAMPLING_TEMPERATURE}" \
    --top-p "${SAMPLING_TOP_P}" \
    --batch-size "${SAMPLING_BATCH_SIZE}"

# ============================================================================
# PHASE 2: LABEL
# ============================================================================

log "=== PHASE 2: Generate labels ==="

run_script "2.0-make-labels.py" \
    --model "${MODEL}" \
    --years ${GENERATION_YEARS[@]} \
    $( [ -n "${GENERATION_MONTH}" ] && echo "--month ${GENERATION_MONTH}" ) \
    --answer-threshold "${ANSWER_THRESHOLD}" \
    --deepinfra-model "${LABEL_MODEL}"

# ============================================================================
# PHASE 3: TRAIN (2025 only)
# ============================================================================

log "=== PHASE 3: Training (train years: ${TRAIN_YEARS[*]}) ==="

run_script "3.0-detection-model-training.py" \
    --train-years "${TRAIN_YEARS[@]}" \
    $( [ -n "${TRAIN_MONTH}" ] && echo "--month ${TRAIN_MONTH}" ) \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}"

run_script "3.1-fit-baselines.py" \
    --train-years "${TRAIN_YEARS[@]}" \
    $( [ -n "${TRAIN_MONTH}" ] && echo "--month ${TRAIN_MONTH}" ) \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}" \
    --num-samples "${NUM_SAMPLES}" \
    --temperature "${SAMPLING_TEMPERATURE}"

run_script "3.2-train-halunet.py" \
    --train-years "${TRAIN_YEARS[@]}" \
    $( [ -n "${TRAIN_MONTH}" ] && echo "--month ${TRAIN_MONTH}" ) \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}"

# ============================================================================
# PHASE 4: EVALUATE (January 2026)
# ============================================================================

log "=== PHASE 4: OOD evaluation (test: ${TEST_YEARS[*]}-$(printf '%02d' ${TEST_MONTH})) ==="

run_script "4.0-model-evaluation.py" \
    --test-years "${TEST_YEARS[@]}" \
    --test-month "${TEST_MONTH}" \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}" \
    --num-samples "${NUM_SAMPLES}"

# ============================================================================
# PHASE 5: RESULTS ANALYSIS
# ============================================================================

log "=== PHASE 5: Results analysis ==="

MODEL_SLUG="${MODEL//\//__}"

run_script "5.0-results-analysis.py" \
    --model "${MODEL}" \
    --test-years "${TEST_YEARS[@]}" \
    $( [ -n "${TEST_MONTH}" ] && echo "--test-month ${TEST_MONTH}" ) \
    --thresholds-file "models/${MODEL_SLUG}/thresholds.json" \
    --plot-dpi 300

log ""
log "=== PIPELINE COMPLETE ==="
log "Predictions in: data/realtimeqa-YYYY(-MM)/<model_slug>/predictions/"
log "Analysis in:  data/realtimeqa-YYYY(-MM)/<model_slug>/analysis/"
