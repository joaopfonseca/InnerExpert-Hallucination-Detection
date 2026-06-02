#!/usr/bin/env bash
# ============================================================================
# train_pipeline.sh — Training pipeline for hallucination detector + baselines.
#
# Generates the training data (RealtimeQA 2025), labels it, and trains:
#   • MoE hallucination detector (3.0)
#   • Baseline thresholds (3.1)
#   • HaluNet (3.2)
#
# Usage:
#   chmod +x train_pipeline.sh
#   ./train_pipeline.sh
#
# Configuration lives in pipeline_config.sh (sourced automatically).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_config.sh"

# Resolve model slug once
MODEL_SLUG=$(derive_model_slug "${MODEL}")
MODEL_DIR="models/${MODEL_SLUG}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }

run_script() {
    local script="$1"; shift
    log "Running: ${script} $*"
    python "${SCRIPT_DIR}/${script}" "$@"
}

# ============================================================================
# PHASE 1 — Generate training data
# ============================================================================

log "=== PHASE 1: Generate answers (train years: ${TRAIN_YEARS[*]}) ==="

if all_years_have_results "${TRAIN_MONTH}" "${TRAIN_YEARS[@]}"; then
    log "SKIP: all target years already have results.parquet — skipping 1.0-generate-answers.py"
else
    run_script "1.0-generate-answers.py" \
        --model "${MODEL}" \
        --quantize "${QUANTIZE}" \
        --years "${TRAIN_YEARS[@]}" \
        $( [[ -n "${TRAIN_MONTH}" ]] && echo "--month ${TRAIN_MONTH}" ) \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        ${RETURN_BASELINE_FEATURES}
fi

log "=== PHASE 1b: Generate baseline samples ==="

if all_years_have_samples "${TRAIN_MONTH}" "${TRAIN_YEARS[@]}"; then
    log "SKIP: all target years already have sampled_generation/*.pt — skipping 1.1-generate-baseline-samples.py"
else
    run_script "1.1-generate-baseline-samples.py" \
        --model "${MODEL}" \
        --quantize "${QUANTIZE}" \
        --years "${TRAIN_YEARS[@]}" \
        $( [[ -n "${TRAIN_MONTH}" ]] && echo "--month ${TRAIN_MONTH}" ) \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --num-samples "${NUM_SAMPLES}" \
        --temperature "${SAMPLING_TEMPERATURE}" \
        --top-p "${SAMPLING_TOP_P}" \
        --batch-size "${SAMPLING_BATCH_SIZE}"
fi

# ============================================================================
# PHASE 2 — Label training data
# ============================================================================

log "=== PHASE 2: Generate labels (train years: ${TRAIN_YEARS[*]}) ==="

if all_years_have_labels "${TRAIN_MONTH}" "${TRAIN_YEARS[@]}"; then
    log "SKIP: all target years already have results_labeled_*.parquet — skipping 2.0-make-labels.py"
else
    run_script "2.0-make-labels.py" \
        --model "${MODEL}" \
        --years "${TRAIN_YEARS[@]}" \
        $( [[ -n "${TRAIN_MONTH}" ]] && echo "--month ${TRAIN_MONTH}" ) \
        --answer-threshold "${ANSWER_THRESHOLD}" \
        --deepinfra-model "${LABEL_MODEL}"
fi

# ============================================================================
# PHASE 3 — Train
# ============================================================================

log "=== PHASE 3: Training ==="

if [[ -f "${MODEL_DIR}/detector.pkl" ]]; then
    log "SKIP: ${MODEL_DIR}/detector.pkl already exists — skipping 3.0-detection-model-training.py"
else
    run_script "3.0-detection-model-training.py" \
        --train-years "${TRAIN_YEARS[@]}" \
        $( [[ -n "${TRAIN_MONTH}" ]] && echo "--month ${TRAIN_MONTH}" ) \
        --model "${MODEL}" \
        --label-model "${LABEL_MODEL}"
fi

if [[ -f "${MODEL_DIR}/thresholds.json" ]]; then
    log "SKIP: ${MODEL_DIR}/thresholds.json already exists — skipping 3.1-fit-baselines.py"
else
    run_script "3.1-fit-baselines.py" \
        --train-years "${TRAIN_YEARS[@]}" \
        $( [[ -n "${TRAIN_MONTH}" ]] && echo "--month ${TRAIN_MONTH}" ) \
        --model "${MODEL}" \
        --label-model "${LABEL_MODEL}" \
        --num-samples "${NUM_SAMPLES}" \
        --temperature "${SAMPLING_TEMPERATURE}"
fi

if [[ -f "${MODEL_DIR}/halunet.pt" ]]; then
    log "SKIP: ${MODEL_DIR}/halunet.pt already exists — skipping 3.2-train-halunet.py"
else
    run_script "3.2-train-halunet.py" \
        --train-years "${TRAIN_YEARS[@]}" \
        $( [[ -n "${TRAIN_MONTH}" ]] && echo "--month ${TRAIN_MONTH}" ) \
        --model "${MODEL}" \
        --label-model "${LABEL_MODEL}"
fi

log ""
log "=== TRAIN PIPELINE COMPLETE ==="
log "Models saved in: ${MODEL_DIR}/"
log "  • detector.pkl"
log "  • thresholds.json"
log "  • halunet.pt"
