#!/usr/bin/env bash
# ===========================================================================
# oos_pipeline.sh — Out-of-sample cross-dataset evaluation pipeline.
#
# Runs the full OOS pipeline for all (or a subset of) OOS datasets:
#   6.0-oos-generate.py   — generate answers + sampled responses
#   6.1-oos-label.py      — LLM-as-judge labeling (GLM-5.1 via DeepInfra)
#   7.0-oos-evaluation.py — run all detection methods, save predictions
#   8.0-oos-analysis.py   — cross-dataset comparison + plots
#
# Requires the trained models from the train pipeline:
#   models/<model_slug>/detector.pkl
#   models/<model_slug>/thresholds.json
#   models/<model_slug>/halunet.pt
#
# Usage:
#   bash experiments/oos_pipeline.sh
#   Datasets: override via OOS_DATASETS="squad truthfulqa" bash experiments/oos_pipeline.sh
#   Skip sampled: SAMPLED=0 bash experiments/oos_pipeline.sh
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Source shared config (MODEL, QUANTIZE, LABEL_MODEL, MAX_NEW_TOKENS, etc.)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pipeline_config.sh"

# OOS-specific config
OOS_DATASETS="${OOS_DATASETS:-squad truthfulqa nq_open freshqa}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SAMPLED="${SAMPLED:-1}"
BATCH_SIZE="${BATCH_SIZE:-6}"

MODEL_SLUG="$(derive_model_slug "${MODEL}")"
MODEL_DIR="models/${MODEL_SLUG}"

log() {
    echo ""
    echo "======================================================================="
    echo "  $*"
    echo "======================================================================="
}

run_script() {
    local script="$1"
    shift
    echo ""
    echo "--- ${script} $* ---"
    python "${SCRIPT_DIR}/${script}" "$@"
}

# ---------------------------------------------------------------------------
# Pre-flight: check trained models exist
# ---------------------------------------------------------------------------
log "OOS PIPELINE — pre-flight checks"

if [[ ! -f "${MODEL_DIR}/detector.pkl" ]]; then
    echo "ERROR: ${MODEL_DIR}/detector.pkl not found. Run train_pipeline.sh first." >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/thresholds.json" ]]; then
    echo "ERROR: ${MODEL_DIR}/thresholds.json not found. Run train_pipeline.sh first." >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/halunet.pt" ]]; then
    echo "WARNING: ${MODEL_DIR}/halunet.pt not found — HaluNet will be skipped in 7.0." >&2
fi

echo "Model:        ${MODEL}"
echo "Datasets:     ${OOS_DATASETS}"
echo "Num samples:  ${NUM_SAMPLES}"
echo "Sampled:      ${SAMPLED}"

# ---------------------------------------------------------------------------
# Per-dataset: 6.0 (generate) + 6.1 (label) + 7.0 (evaluate)
# ---------------------------------------------------------------------------
for DS in ${OOS_DATASETS}; do
    log "DATASET: ${DS}"

    DATA_DIR="data/oos-${DS}/${MODEL_SLUG}"
    RESULTS_FILE="${DATA_DIR}/results.parquet"
    LABEL_FILE="${DATA_DIR}/results_labeled_$(echo "${LABEL_MODEL}" | tr '/' '__').parquet"
    PREDICTIONS_DIR="${DATA_DIR}/predictions"

    # --- 6.0: Generate ---
    if [[ -f "${RESULTS_FILE}" ]]; then
        log "SKIP 6.0: ${RESULTS_FILE} already exists"
    else
        SAMPLED_FLAGS=""
        if [[ "${SAMPLED}" == "1" ]]; then
            SAMPLED_FLAGS="--num-samples ${NUM_SAMPLES}"
        fi
        run_script "6.0-oos-generate.py" \
            --dataset "${DS}" \
            --model "${MODEL}" \
            --quantize "${QUANTIZE}" \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --batch-size "${BATCH_SIZE}" \
            ${SAMPLED_FLAGS}
    fi

    # --- 6.1: Label ---
    if [[ -f "${LABEL_FILE}" ]]; then
        log "SKIP 6.1: ${LABEL_FILE} already exists"
    else
        run_script "6.1-oos-label.py" \
            --dataset "${DS}" \
            --model "${MODEL}" \
            --label-model "${LABEL_MODEL}"
    fi

    # --- 7.0: Evaluate ---
    if [[ -d "${PREDICTIONS_DIR}" && -f "${PREDICTIONS_DIR}/ground_truth.parquet" ]]; then
        log "SKIP 7.0: ${PREDICTIONS_DIR}/ground_truth.parquet already exists"
    else
        SAMPLED_FLAGS=""
        if [[ "${SAMPLED}" != "1" ]]; then
            SAMPLED_FLAGS="--skip-sampled"
        fi
        run_script "7.0-oos-evaluation.py" \
            --dataset "${DS}" \
            --model "${MODEL}" \
            --label-model "${LABEL_MODEL}" \
            --models-dir "${MODEL_DIR}" \
            --num-samples "${NUM_SAMPLES}" \
            ${SAMPLED_FLAGS}
    fi
done

# ---------------------------------------------------------------------------
# 8.0: Cross-dataset analysis
# ---------------------------------------------------------------------------
log "8.0 — CROSS-DATASET ANALYSIS"

COMPARISON_DIR="data/oos-comparison"
if [[ -f "${COMPARISON_DIR}/results.json" ]]; then
    log "SKIP 8.0: ${COMPARISON_DIR}/results.json already exists"
else
    run_script "8.0-oos-analysis.py" \
        --datasets ${OOS_DATASETS} \
        --model "${MODEL}" \
        --models-dir "${MODEL_DIR}" \
        --thresholds-file "${MODEL_DIR}/thresholds.json" \
        --output-dir "${COMPARISON_DIR}"
fi

log "OOS PIPELINE COMPLETE"
echo "Results: ${COMPARISON_DIR}/"