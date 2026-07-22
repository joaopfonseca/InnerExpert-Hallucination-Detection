#!/usr/bin/env bash
# ===========================================================================
# rerun_individual_signals.sh — Fit + evaluate individual MoE signals
#
# Runs the full pipeline for individual MoE signal evaluation:
#   1. 3.3-fit-individual-signals.py — fit thresholds + best layers on train
#   2. 4.0-model-evaluation.py — re-evaluate RTQA (regenerates signal parquets)
#   3. 6.2-oos-evaluation.py — re-evaluate OOS datasets (regenerates signal parquets)
#   4. 5.0-results-analysis.py — re-run RTQA analysis with --thresholds-file
#   5. 6.3-oos-analysis.py — re-run OOS analysis with --thresholds-file
#
# No model inference needed — 4.0/6.2 load pre-computed outputs from .pt files.
# Estimated time: ~1-2 hours for both models + all datasets.
#
# Usage:
#   bash experiments/rerun_individual_signals.sh
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "${SCRIPT_DIR}")"

source "${SCRIPT_DIR}/pipeline_config.sh"

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
# Step 1: Fit individual signal thresholds on train data (both models)
# ===========================================================================
log "STEP 1/5: Fit individual signal thresholds"

for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    log "Fitting thresholds / ${MODEL_SLUG}"

    run python experiments/3.3-fit-individual-signals.py \
        --model "${MODEL}" \
        --train-years 2024 2025 \
        --label-model "${LABEL_MODEL}"
done

# ===========================================================================
# Step 2: Re-run 4.0 (RTQA eval) — regenerate all signal parquets
# ===========================================================================
log "STEP 2/5: RTQA 2026 — re-evaluate (regenerate signal parquets)"

for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    log "RTQA 2026 / ${MODEL_SLUG}"

    run python experiments/4.0-model-evaluation.py \
        --model "${MODEL}" \
        --test-years 2026 \
        --label-model "${LABEL_MODEL}" \
        --models-dir models \
        --num-samples 5
done

# ===========================================================================
# Step 3: Re-run 6.2 (OOS eval) — regenerate all signal parquets
# ===========================================================================
log "STEP 3/5: OOS — re-evaluate (regenerate signal parquets)"

for DS in squad truthfulqa nq_open freshqa; do
    for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
        MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
        log "${DS} / ${MODEL_SLUG}"

        run python experiments/6.2-oos-evaluation.py \
            --dataset "${DS}" \
            --model "${MODEL}" \
            --label-model "${LABEL_MODEL}" \
            --models-dir models \
            --num-samples 5
    done
done

# ===========================================================================
# Step 4: Re-run 5.0 (RTQA analysis) with --thresholds-file
# ===========================================================================
log "STEP 4/5: RTQA 2026 — re-run analysis with --thresholds-file"

for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    MODEL_DIR="models/${MODEL_SLUG}"
    THRESHOLDS_FILE="${MODEL_DIR}/thresholds.json"
    log "RTQA 2026 analysis / ${MODEL_SLUG}"

    if [[ ! -f "${THRESHOLDS_FILE}" ]]; then
        echo "ERROR: ${THRESHOLDS_FILE} not found. Skipping ${MODEL_SLUG}."
        continue
    fi

    rm -rf "data/realtimeqa-2026/${MODEL_SLUG}/analysis"

    run python experiments/5.0-results-analysis.py \
        --model "${MODEL}" \
        --test-years 2026 \
        --thresholds-file "${THRESHOLDS_FILE}" \
        --plot-dpi 300
done

# ===========================================================================
# Step 5: Re-run 6.3 (OOS analysis) with --thresholds-file
# ===========================================================================
log "STEP 5/5: OOS — re-run cross-dataset analysis with --thresholds-file"

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

    rm -rf "${COMPARISON_DIR}"

    run python experiments/6.3-oos-analysis.py \
        --datasets squad truthfulqa nq_open freshqa \
        --model "${MODEL}" \
        --models-dir "${MODEL_DIR}" \
        --thresholds-file "${THRESHOLDS_FILE}" \
        --output-dir "${COMPARISON_DIR}"
done

log "ALL DONE — rerun_individual_signals.sh complete"
echo ""
echo "Updated outputs:"
echo "  Thresholds:       models/{model_slug}/thresholds.json"
echo "  RTQA predictions:  data/realtimeqa-2026/{model_slug}/predictions/signal_*.parquet"
echo "  RTQA analysis:    data/realtimeqa-2026/{model_slug}/analysis/"
echo "  OOS predictions:  data/oos-{ds}/{model_slug}/predictions/signal_*.parquet"
echo "  OOS comparison:   data/oos-comparison/{model_slug}/"
echo ""
echo "Sync back to local with:"
echo "  rsync -avz nutrina:~/MoE-Uncertainty-Estimation/data/oos-comparison/ data/oos-comparison/"
echo "  rsync -avz --include='*/' --include='analysis/**' --exclude='*' \\"
echo "    nutrina:~/MoE-Uncertainty-Estimation/data/realtimeqa-2026/ data/realtimeqa-2026/"