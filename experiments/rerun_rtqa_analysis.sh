#!/usr/bin/env bash
# ===========================================================================
# rerun_rtqa_analysis.sh — Re-run 5.0-results-analysis.py with --thresholds-file
#
# Fixes the bug in fix_sampled_evidence.sh where step 7 called 5.0 without
# --thresholds-file, causing all methods to fall back to threshold=0.5.
# This produced F1=0 for LLM-Check and SemanticEnergy whose score ranges
# are entirely below 0.5.
#
# Usage:
#   bash experiments/rerun_rtqa_analysis.sh
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
# Re-run RTQA 2026 analysis with --thresholds-file (both models)
# ===========================================================================
log "Re-running RTQA 2026 analysis with --thresholds-file"

for MODEL_SLUG in allenai__OLMoE-1B-7B-0924-Instruct google__gemma-4-26B-A4B-it; do
    MODEL=$(echo "${MODEL_SLUG}" | sed 's|__|/|')
    MODEL_DIR="models/${MODEL_SLUG}"
    THRESHOLDS_FILE="${MODEL_DIR}/thresholds.json"

    log "RTQA 2026 / ${MODEL_SLUG}"

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

log "ALL DONE — rerun_rtqa_analysis.sh complete"
echo ""
echo "Updated outputs:"
echo "  data/realtimeqa-2026/{model_slug}/analysis/"
echo ""
echo "Sync back to local with:"
echo "  rsync -avz --include='*/' --include='analysis/**' --exclude='*' \\"
echo "    nutrina:~/MoE-Uncertainty-Estimation/data/realtimeqa-2026/ data/realtimeqa-2026/"