#!/usr/bin/env bash
# ============================================================================
# End-to-end pipeline runner for MoE Uncertainty Estimation experiments
#
# Train on 2022-2025, test OOD on January 2026.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Override variables:
#   MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" ./run_pipeline.sh
# ============================================================================

set -euo pipefail

# --- PHASE 1: GENERATION (run once per dataset) ---
# These generate answers for ALL data (train + test years).
# Training scripts later filter to their --train-years subset.
# 4.0 evaluation filters to its --test-years/--test-month subset.

GENERATION_YEARS=(2022 2023 2024 2025 2026)
GENERATION_MONTH=""             # Empty = all months for each year

# --- TRAINING CONFIGURATION ---
TRAIN_YEARS=(2022 2023 2024 2025)
TRAIN_MONTH=""                  # Empty = all months for each training year

# --- TEST CONFIGURATION ---
TEST_YEARS=(2026)
TEST_MONTH=1                    # January 2026

# --- MODEL CONFIGURATION ---
MODEL="allenai/OLMoE-1B-7B-0924-Instruct"
QUANTIZE="4-bit"                # "16-bit", "8-bit", or "4-bit"
LABEL_MODEL="zai-org/GLM-5.1"   # LLM-as-a-Judge model for labeling

# --- GENERATION CONFIGURATION ---
MAX_NEW_TOKENS=65

# --- SAMPLING CONFIGURATION ---
NUM_SAMPLES=5
SAMPLING_TEMPERATURE=0.7
SAMPLING_TOP_P=0.9
SAMPLING_BATCH_SIZE=2

# --- LABELING CONFIGURATION ---
ANSWER_THRESHOLD=0.5

# --- PATHS ---
# Must be true for HaluNet (needs log_likelihoods + entropies)
# and LLM-Check perplexity (pre-computed by compute_metrics).
RETURN_BASELINE_FEATURES="--return-baseline-features"

# --- PATHS ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"

# ============================================================================
# Helper functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"
}

run_script() {
    local script="$1"
    shift
    log "Running: ${script} $*"
    "${VENV_PYTHON}" "${SCRIPT_DIR}/${script}" "$@"
}

# ============================================================================
# PHASE 1: GENERATE DATA
# ============================================================================

log ""
log "============================================================"
log "PHASE 1: GENERATE ANSWERS"
log "============================================================"
log ""

log "--- 1.0 Generate answers (single-pass, greedy) ---"
run_script "1.0-generate-answers.py" \
    --model "${MODEL}" \
    --quantize "${QUANTIZE}" \
    --years "${GENERATION_YEARS[@]}" \
    $( [ -n "${GENERATION_MONTH}" ] && echo "--month ${GENERATION_MONTH}" ) \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    ${RETURN_BASELINE_FEATURES}

log "--- 1.1 Generate sampled responses (for SU, SE, SelfCheckGPT) ---"
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
# PHASE 2: GENERATE LABELS
# ============================================================================

log ""
log "============================================================"
log "PHASE 2: GENERATE LABELS"
log "============================================================"
log ""

LABEL_YEARS="${GENERATION_YEARS[@]}"   # Label all generated data

run_script "2.0-make-labels.py" \
    --model "${MODEL}" \
    --years ${LABEL_YEARS} \
    $( [ -n "${GENERATION_MONTH}" ] && echo "--month ${GENERATION_MONTH}" ) \
    --answer-threshold "${ANSWER_THRESHOLD}" \
    --deepinfra-model "${LABEL_MODEL}"

# ============================================================================
# PHASE 3: TRAINING (all on 2022-2025)
# ============================================================================

log ""
log "============================================================"
log "PHASE 3: TRAINING (train years: ${TRAIN_YEARS[*]})"
log "============================================================"
log ""

log "--- 3.0 Train MoE detection model ---"
run_script "3.0-detection-model-training.py" \
    --train-years "${TRAIN_YEARS[@]}" \
    $( [ -n "${TRAIN_MONTH}" ] && echo "--month ${TRAIN_MONTH}" ) \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}"

log "--- 3.1 Fit baseline thresholds ---"
run_script "3.1-fit-baselines.py" \
    --train-years "${TRAIN_YEARS[@]}" \
    $( [ -n "${TRAIN_MONTH}" ] && echo "--month ${TRAIN_MONTH}" ) \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}" \
    --num-samples "${NUM_SAMPLES}" \
    --temperature "${SAMPLING_TEMPERATURE}"

log "--- 3.2 Train HaluNet ---"
run_script "3.2-train-halunet.py" \
    --train-years "${TRAIN_YEARS[@]}" \
    $( [ -n "${TRAIN_MONTH}" ] && echo "--month ${TRAIN_MONTH}" ) \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}"

# ============================================================================
# PHASE 4: OOD EVALUATION (on January 2026)
# ============================================================================

log ""
log "============================================================"
log "PHASE 4: OOD EVALUATION (test: ${TEST_YEARS[*]}-$(printf '%02d' ${TEST_MONTH}))"
log "============================================================"
log ""

run_script "4.0-model-evaluation.py" \
    --test-years "${TEST_YEARS[@]}" \
    --test-month "${TEST_MONTH}" \
    --model "${MODEL}" \
    --label-model "${LABEL_MODEL}" \
    --num-samples "${NUM_SAMPLES}"

# ============================================================================
# DONE
# ============================================================================

log ""
log "============================================================"
log "PIPELINE COMPLETE"
log "============================================================"
log ""
log "Results:"
log "  Generated answers:    data/realtimeqa-YYYY(-MM)/<model_slug>/{base,evidence}_generation/"
log "  Sampled responses:    data/realtimeqa-YYYY(-MM)/<model_slug>/sampled_generation/"
log "  Labels:               data/realtimeqa-YYYY(-MM)/<model_slug>/results_labeled_*.parquet"
log "  Trained models:       models/<model_slug>/detector.pkl, halunet.pt, thresholds.json"
log "  Predictions:          data/realtimeqa-YYYY(-MM)/<model_slug>/predictions/"
log ""
log "Next step: python experiments/5.0-results-analysis.py"
