#!/usr/bin/env bash
# ===========================================================================
# rerun_labels.sh — Re-label + re-evaluate datasets with answer_str as evidence
#
# Fixes the judge prompt for non-evidence datasets (TruthfulQA, NQ-Open,
# FreshQA) and partially-evidence datasets (RTQA base rows without context).
# Previously the evidence field was empty, so the judge had no reference to
# compare the generated answer against — leading to massive false positives.
# Now the ground-truth answer_str is used as the evidence field when the
# context passage is missing.
#
# Steps:
#   1. Delete old labeled parquets (TruthfulQA, NQ-Open, FreshQA, RTQA)
#   2. Re-label OOS via 6.1-oos-label.py (3 datasets × 2 models)
#   3. Re-label RTQA via 2.0-make-labels.py (2 models)
#   4. Re-evaluate OOS via 6.2-oos-evaluation.py (3 datasets × 2 models)
#   5. Re-evaluate RTQA via 4.0-model-evaluation.py (2 models)
#   6. Re-analyze OOS via 6.3-oos-analysis.py (both models)
#   7. Re-analyze RTQA via 5.0-results-analysis.py (both models)
#
# SQuAD is NOT re-labeled (it already has context passages).
# No retraining needed — detector/thresholds are unchanged.
#
# Requires DEEPINFRA_API_KEY in .env for LLM-as-judge calls.
# Estimated time: ~2-3 hours (API calls + re-evaluation).
#
# Usage:
#   bash experiments/rerun_labels.sh
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

MODELS=(
    "allenai__OLMoE-1B-7B-0924-Instruct"
    "google__gemma-4-26B-A4B-it"
)

# ===========================================================================
# Step 1: Delete old labeled parquets
# ===========================================================================
log "STEP 1/7: Delete old labeled parquets"

for MODEL_SLUG in "${MODELS[@]}"; do
    for DS in truthfulqa nq_open freshqa; do
        LABEL_FILE="data/oos-${DS}/${MODEL_SLUG}/results_labeled_${LABEL_MODEL_SLUG}.parquet"
        if [[ -f "${LABEL_FILE}" ]]; then
            rm -f "${LABEL_FILE}"
            echo "  Deleted ${LABEL_FILE}"
        fi
        # Also delete predictions (will be regenerated)
        rm -rf "data/oos-${DS}/${MODEL_SLUG}/predictions"
    done
    # RTQA
    LABEL_FILE="data/realtimeqa-2026/${MODEL_SLUG}/results_labeled_${LABEL_MODEL_SLUG}.parquet"
    if [[ -f "${LABEL_FILE}" ]]; then
        rm -f "${LABEL_FILE}"
        echo "  Deleted ${LABEL_FILE}"
    fi
    rm -rf "data/realtimeqa-2026/${MODEL_SLUG}/predictions"
done

# ===========================================================================
# Step 2: Re-label OOS datasets via 6.1-oos-label.py
# ===========================================================================
log "STEP 2/7: Re-label OOS datasets (6.1-oos-label.py)"

for DS in truthfulqa nq_open freshqa; do
    for MODEL_SLUG in "${MODELS[@]}"; do
        MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
        log "Labeling ${DS} / ${MODEL_SLUG}"

        run python experiments/6.1-oos-label.py \
            --dataset "${DS}" \
            --model "${MODEL}" \
            --label-model "${LABEL_MODEL}"
    done
done

# ===========================================================================
# Step 3: Re-label RTQA 2026 via 2.0-make-labels.py
# ===========================================================================
log "STEP 3/7: Re-label RTQA 2026 (2.0-make-labels.py)"

for MODEL_SLUG in "${MODELS[@]}"; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    log "Labeling RTQA 2026 / ${MODEL_SLUG}"

    run python experiments/2.0-make-labels.py \
        --model "${MODEL}" \
        --years 2026 \
        --answer-threshold "${ANSWER_THRESHOLD}" \
        --deepinfra-model "${LABEL_MODEL}"
done

# ===========================================================================
# Step 4: Re-evaluate OOS via 6.2-oos-evaluation.py
# ===========================================================================
log "STEP 4/7: Re-evaluate OOS datasets (6.2-oos-evaluation.py)"

for DS in truthfulqa nq_open freshqa; do
    for MODEL_SLUG in "${MODELS[@]}"; do
        MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
        log "Evaluating ${DS} / ${MODEL_SLUG}"

        run python experiments/6.2-oos-evaluation.py \
            --dataset "${DS}" \
            --model "${MODEL}" \
            --label-model "${LABEL_MODEL}" \
            --models-dir models \
            --num-samples 5
    done
done

# ===========================================================================
# Step 5: Re-evaluate RTQA via 4.0-model-evaluation.py
# ===========================================================================
log "STEP 5/7: Re-evaluate RTQA 2026 (4.0-model-evaluation.py)"

for MODEL_SLUG in "${MODELS[@]}"; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    log "Evaluating RTQA 2026 / ${MODEL_SLUG}"

    run python experiments/4.0-model-evaluation.py \
        --model "${MODEL}" \
        --test-years 2026 \
        --label-model "${LABEL_MODEL}" \
        --models-dir models \
        --num-samples 5
done

# ===========================================================================
# Step 6: Re-analyze OOS via 6.3-oos-analysis.py
# ===========================================================================
log "STEP 6/7: Re-analyze OOS (6.3-oos-analysis.py)"

for MODEL_SLUG in "${MODELS[@]}"; do
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

# ===========================================================================
# Step 7: Re-analyze RTQA via 5.0-results-analysis.py
# ===========================================================================
log "STEP 7/7: Re-analyze RTQA 2026 (5.0-results-analysis.py)"

for MODEL_SLUG in "${MODELS[@]}"; do
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

log "ALL DONE — rerun_labels.sh complete"
echo ""
echo "Updated outputs:"
echo "  OOS labels:       data/oos-{truthfulqa,nq_open,freshqa}/{model_slug}/results_labeled_*.parquet"
echo "  RTQA labels:      data/realtimeqa-2026/{model_slug}/results_labeled_*.parquet"
echo "  OOS predictions:  data/oos-{ds}/{model_slug}/predictions/"
echo "  RTQA predictions: data/realtimeqa-2026/{model_slug}/predictions/"
echo "  OOS comparison:   data/oos-comparison/{model_slug}/"
echo "  RTQA analysis:    data/realtimeqa-2026/{model_slug}/analysis/"
echo ""
echo "Sync back to local with:"
echo "  rsync -avz nutrina:~/MoE-Uncertainty-Estimation/data/oos-comparison/ data/oos-comparison/"
echo "  rsync -avz --include='*/' --include='analysis/**' --exclude='*' \\"
echo "    nutrina:~/MoE-Uncertainty-Estimation/data/realtimeqa-2026/ data/realtimeqa-2026/"