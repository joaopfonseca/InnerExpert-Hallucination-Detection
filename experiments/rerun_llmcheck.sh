#!/usr/bin/env bash
# ===========================================================================
# rerun_llmcheck.sh — Re-run LLM-Check evaluation with best-layer scores
#
# Fixes the layer mismatch bug where 3.1-fit-baselines.py fit thresholds on
# a single best layer, but predictions.py averaged across all layers.  Now
# predictions.py reads the best layer from thresholds.json and slices to
# that layer, matching 3.1's convention.
#
# This script re-runs:
#   1. 4.0 (RTQA eval) — regenerates llm_check.parquet with best-layer scores
#   2. 6.2 (OOS eval) for each dataset — same
#   3. 5.0 (RTQA analysis) with --thresholds-file
#   4. 6.3 (OOS analysis) with --thresholds-file
#
# No model inference needed — 4.0/6.2 load pre-computed outputs from .pt files.
# Estimated time: ~1-2 hours for both models + all datasets.
#
# Usage:
#   bash experiments/rerun_llmcheck.sh
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "${SCRIPT_DIR}")"

source "${SCRIPT_DIR}/pipeline_config.sh"

LABEL_MODEL_SLUG="$(echo "${LABEL_MODEL}" | sed 's|/|__|g')"

log() {
    echo ""
    echo "======================================================================="
    echo "  $*"
    echo "======================================================================="
}

run() {
    echo ""
    echo "--- $* ---"
    "$@"
}

# ===========================================================================
# Step 1: Re-run 4.0 (RTQA eval) — regenerate llm_check.parquet
# ===========================================================================
log "STEP 1/4: RTQA 2026 — re-evaluate LLM-Check (best-layer scores)"

for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    PRED_DIR="data/realtimeqa-2026/${MODEL_SLUG}/predictions"
    log "RTQA 2026 / ${MODEL_SLUG}"

    # Delete old llm_check.parquet so 4.0 regenerates it
    rm -f "${PRED_DIR}/llm_check.parquet"

    run python experiments/4.0-model-evaluation.py \
        --model "${MODEL}" \
        --test-years 2026 \
        --label-model "${LABEL_MODEL}" \
        --models-dir models \
        --num-samples 5
done

# ===========================================================================
# Step 2: Re-run 6.2 (OOS eval) — regenerate llm_check.parquet per dataset
# ===========================================================================
log "STEP 2/4: OOS — re-evaluate LLM-Check (best-layer scores)"

for DS in squad truthfulqa nq_open freshqa; do
    for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
        MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
        PRED_DIR="data/oos-${DS}/${MODEL_SLUG}/predictions"
        log "${DS} / ${MODEL_SLUG}"

        # Delete old llm_check.parquet so 6.2 regenerates it
        rm -f "${PRED_DIR}/llm_check.parquet"

        run python experiments/6.2-oos-evaluation.py \
            --dataset "${DS}" \
            --model "${MODEL}" \
            --label-model "${LABEL_MODEL}" \
            --models-dir models \
            --num-samples 5
    done
done

# ===========================================================================
# Step 3: Re-run 5.0 (RTQA analysis) with --thresholds-file
# ===========================================================================
log "STEP 3/4: RTQA 2026 — re-run analysis with --thresholds-file"

for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    MODEL_DIR="models/${MODEL_SLUG}"
    THRESHOLDS_FILE="${MODEL_DIR}/thresholds.json"
    log "RTQA 2026 analysis / ${MODEL_SLUG}"

    if [[ ! -f "${THRESHOLDS_FILE}" ]]; then
        echo "ERROR: ${THRESHOLDS_FILE} not found. Skipping ${MODEL_SLUG}."
        continue
    fi

    # Remove old analysis outputs so 5.0 doesn't skip
    rm -rf "data/realtimeqa-2026/${MODEL_SLUG}/analysis"

    run python experiments/5.0-results-analysis.py \
        --model "${MODEL}" \
        --test-years 2026 \
        --thresholds-file "${THRESHOLDS_FILE}" \
        --plot-dpi 300
done

# ===========================================================================
# Step 4: Re-run 6.3 (OOS analysis) with --thresholds-file
# ===========================================================================
log "STEP 4/4: OOS — re-run cross-dataset analysis with --thresholds-file"

for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    MODEL_DIR="models/${MODEL_SLUG}"
    THRESHOLDS_FILE="${MODEL_DIR}/thresholds.json"
    COMPARISON_DIR="data/oos-comparison/${MODEL_SLUG}"
    log "OOS analysis / ${MODEL_SLUG}"

    if [[ ! -f "${THRESHOLDS_FILE}" ]]; then
        echo "ERROR: ${THRESHOLDS_FILE} not found. Skipping ${MODEL_SLUG}."
        continue
    fi

    # Remove old comparison outputs so 6.3 regenerates
    rm -rf "${COMPARISON_DIR}"

    run python experiments/6.3-oos-analysis.py \
        --datasets squad truthfulqa nq_open freshqa \
        --model "${MODEL}" \
        --models-dir "${MODEL_DIR}" \
        --thresholds-file "${THRESHOLDS_FILE}" \
        --output-dir "${COMPARISON_DIR}"
done

log "ALL DONE — rerun_llmcheck.sh complete"
echo ""
echo "Updated outputs:"
echo "  RTQA predictions: data/realtimeqa-2026/{model_slug}/predictions/llm_check.parquet"
echo "  RTQA analysis:    data/realtimeqa-2026/{model_slug}/analysis/"
echo "  OOS predictions:  data/oos-{ds}/{model_slug}/predictions/llm_check.parquet"
echo "  OOS comparison:   data/oos-comparison/{model_slug}/"
echo ""
echo "Sync back to local with:"
echo "  rsync -avz nutrina:~/MoE-Uncertainty-Estimation/data/oos-comparison/ data/oos-comparison/"
echo "  rsync -avz --include='*/' --include='analysis/**' --exclude='*' \\"
echo "    nutrina:~/MoE-Uncertainty-Estimation/data/realtimeqa-2026/ data/realtimeqa-2026/"