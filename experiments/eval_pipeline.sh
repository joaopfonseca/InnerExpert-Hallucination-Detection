#!/usr/bin/env bash
# ============================================================================
# eval_pipeline.sh — OOD evaluation pipeline for the hallucination detector.
#
# Generates OOD test data (RealtimeQA 2026 by default), labels it, runs every
# detection method, and produces comparison plots / tables.
#
# Usage:
#   chmod +x eval_pipeline.sh
#   ./eval_pipeline.sh
#
# Requires train_pipeline.sh to have been run first so that the trained
# artefacts exist in models/<model_slug>/.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_config.sh"

# Resolve model slug once
MODEL_SLUG=$(derive_model_slug "${MODEL}")
MODEL_DIR="models/${MODEL_SLUG}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }

# ============================================================================
# Helpers
# ============================================================================

run_script() {
    local script="$1"; shift
    log "Running: ${script} $*"
    python "${SCRIPT_DIR}/${script}" "$@"
}

# ============================================================================
# PHASE 0 — Pre-flight checks (abort on missing train artefacts)
# ============================================================================

log "=== PHASE 0: Pre-flight checks ==="

missing=0
if [[ ! -f "${MODEL_DIR}/detector.pkl" ]]; then
    log "MISSING: ${MODEL_DIR}/detector.pkl — run train_pipeline.sh first."
    missing=1
fi
if [[ ! -f "${MODEL_DIR}/thresholds.json" ]]; then
    log "MISSING: ${MODEL_DIR}/thresholds.json — run train_pipeline.sh first."
    missing=1
fi
if [[ ! -f "${MODEL_DIR}/halunet.pt" ]]; then
    log "MISSING: ${MODEL_DIR}/halunet.pt — run train_pipeline.sh first."
    missing=1
fi

if [[ ${missing} -eq 1 ]]; then
    log "ABORT: Evaluation pipeline cannot run without trained artefacts."
    exit 1
fi

log "Pre-flight OK — all trained artefacts found."

# ============================================================================
# PHASE 1 — Generate OOD data
# ============================================================================

log "=== PHASE 1: Generate answers (test years: ${TEST_YEARS[*]}) ==="

if all_years_have_results "${TEST_MONTH}" "${TEST_YEARS[@]}"; then
    log "SKIP: all target years already have results.parquet — skipping 1.0-generate-answers.py"
else
    run_script "1.0-generate-answers.py" \
        --model "${MODEL}" \
        --quantize "${QUANTIZE}" \
        --years "${TEST_YEARS[@]}" \
        $( [[ -n "${TEST_MONTH}" ]] && echo "--month ${TEST_MONTH}" ) \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        ${RETURN_BASELINE_FEATURES}
fi

log "=== PHASE 1b: Generate baseline samples ==="

if all_years_have_samples "${TEST_MONTH}" "${TEST_YEARS[@]}"; then
    log "SKIP: all target years already have sampled_generation/*.pt — skipping 1.1-generate-baseline-samples.py"
else
    run_script "1.1-generate-baseline-samples.py" \
        --model "${MODEL}" \
        --quantize "${QUANTIZE}" \
        --years "${TEST_YEARS[@]}" \
        $( [[ -n "${TEST_MONTH}" ]] && echo "--month ${TEST_MONTH}" ) \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --num-samples "${NUM_SAMPLES}" \
        --temperature "${SAMPLING_TEMPERATURE}" \
        --top-p "${SAMPLING_TOP_P}" \
        --batch-size "${SAMPLING_BATCH_SIZE}"
fi

# ============================================================================
# PHASE 2 — Label OOD data
# ============================================================================

log "=== PHASE 2: Generate labels (test years: ${TEST_YEARS[*]}) ==="

if all_years_have_labels "${TEST_MONTH}" "${TEST_YEARS[@]}"; then
    log "SKIP: all target years already have results_labeled_*.parquet — skipping 2.0-make-labels.py"
else
    run_script "2.0-make-labels.py" \
        --model "${MODEL}" \
        --years "${TEST_YEARS[@]}" \
        $( [[ -n "${TEST_MONTH}" ]] && echo "--month ${TEST_MONTH}" ) \
        --answer-threshold "${ANSWER_THRESHOLD}" \
        --deepinfra-model "${LABEL_MODEL}"
fi

# ============================================================================
# PHASE 3 — Evaluate (reads models/<slug>/detector.pkl + thresholds.json)
# ============================================================================

log "=== PHASE 3: OOD evaluation ==="

if all_years_have_predictions "${TEST_MONTH}" "${TEST_YEARS[@]}"; then
    log "SKIP: all target years already have predictions/ — skipping 4.0-model-evaluation.py"
else
    run_script "4.0-model-evaluation.py" \
        --test-years "${TEST_YEARS[@]}" \
        $( [[ -n "${TEST_MONTH}" ]] && echo "--test-month ${TEST_MONTH}" ) \
        --model "${MODEL}" \
        --label-model "${LABEL_MODEL}" \
        --num-samples "${NUM_SAMPLES}"
fi

# ============================================================================
# PHASE 4 — Results analysis
# ============================================================================

log "=== PHASE 4: Results analysis ==="

if all_years_have_analysis "${TEST_MONTH}" "${TEST_YEARS[@]}"; then
    log "SKIP: all target years already have analysis/ — skipping 5.0-results-analysis.py"
else
    run_script "5.0-results-analysis.py" \
        --model "${MODEL}" \
        --test-years "${TEST_YEARS[@]}" \
        $( [[ -n "${TEST_MONTH}" ]] && echo "--test-month ${TEST_MONTH}" ) \
        --thresholds-file "${MODEL_DIR}/thresholds.json" \
        --plot-dpi 300
fi

log ""
log "=== EVAL PIPELINE COMPLETE ==="
log "Predictions in: data/realtimeqa-<year>${TEST_MONTH:+-$(printf '%02d' ${TEST_MONTH})}/<model_slug>/predictions/"
log "Analysis in:    data/realtimeqa-<year>${TEST_MONTH:+-$(printf '%02d' ${TEST_MONTH})}/<model_slug>/analysis/"
