#!/bin/bash
# =============================================================================
# submit_all.sh — Submit the full experiment pipeline to PBS
# =============================================================================
# Usage:
#   bash scripts/submit_all.sh            # Train both + evaluate after
#   bash scripts/submit_all.sh --baseline # Train baseline only
#   bash scripts/submit_all.sh --improved # Train improved only
#   bash scripts/submit_all.sh --eval     # Evaluate only (checkpoints must exist)
#
# This script submits jobs with dependencies so evaluation runs automatically
# after both training jobs complete.
#
# Pipeline:
#   train_baseline.pbs ──┐
#                        ├── evaluate.pbs (runs after both finish)
#   train_improved.pbs ──┘
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

# Parse arguments
RUN_BASELINE=false
RUN_IMPROVED=false
RUN_EVAL=false

if [ $# -eq 0 ]; then
    # No arguments: run everything
    RUN_BASELINE=true
    RUN_IMPROVED=true
    RUN_EVAL=true
else
    for arg in "$@"; do
        case "${arg}" in
            --baseline) RUN_BASELINE=true ;;
            --improved) RUN_IMPROVED=true ;;
            --eval)     RUN_EVAL=true ;;
            --help|-h)
                echo "Usage: bash scripts/submit_all.sh [--baseline] [--improved] [--eval]"
                echo ""
                echo "  No arguments: submit baseline + improved + evaluation pipeline"
                echo "  --baseline:   submit baseline training only"
                echo "  --improved:   submit improved training only"
                echo "  --eval:       submit evaluation only (checkpoints must exist)"
                exit 0
                ;;
            *)
                echo "Unknown argument: ${arg}"
                echo "Use --help for usage information."
                exit 1
                ;;
        esac
    done
fi

# Create logs directory
mkdir -p "${PROJECT_DIR}/logs"

echo "============================================="
echo " SCP-M GAN Experiment Pipeline"
echo " Project: ${PROJECT_DIR}"
echo "============================================="

# Check that virtual environment exists
if [ ! -d "${PROJECT_DIR}/venv311" ]; then
    echo "ERROR: Virtual environment not found at ${PROJECT_DIR}/venv311"
    echo "Run 'bash scripts/setup_env.sh' first."
    exit 1
fi

DEPEND_JOBS=""

# --- Submit baseline training ---
if [ "${RUN_BASELINE}" = true ]; then
    BASELINE_JOB=$(qsub scripts/train_baseline.pbs)
    BASELINE_ID=$(echo "${BASELINE_JOB}" | grep -oP '^\d+' || echo "${BASELINE_JOB}")
    echo "Submitted: train_baseline.pbs  ->  Job ID: ${BASELINE_ID}"
    DEPEND_JOBS="${BASELINE_ID}"
fi

# --- Submit improved training ---
if [ "${RUN_IMPROVED}" = true ]; then
    IMPROVED_JOB=$(qsub scripts/train_improved.pbs)
    IMPROVED_ID=$(echo "${IMPROVED_JOB}" | grep -oP '^\d+' || echo "${IMPROVED_JOB}")
    echo "Submitted: train_improved.pbs  ->  Job ID: ${IMPROVED_ID}"
    if [ -n "${DEPEND_JOBS}" ]; then
        DEPEND_JOBS="${DEPEND_JOBS}:${IMPROVED_ID}"
    else
        DEPEND_JOBS="${IMPROVED_ID}"
    fi
fi

# --- Submit evaluation (with dependency on training jobs) ---
if [ "${RUN_EVAL}" = true ]; then
    if [ -n "${DEPEND_JOBS}" ]; then
        EVAL_JOB=$(qsub -W depend=afterok:${DEPEND_JOBS} scripts/evaluate.pbs)
        echo "Submitted: evaluate.pbs       ->  Job ID: ${EVAL_JOB}"
        echo "  (will run after jobs: ${DEPEND_JOBS})"
    else
        EVAL_JOB=$(qsub scripts/evaluate.pbs)
        echo "Submitted: evaluate.pbs       ->  Job ID: ${EVAL_JOB}"
    fi
fi

echo ""
echo "============================================="
echo " Monitor jobs: qstat -u \$USER"
echo " Cancel a job: qdel <JOBID>"
echo " View output:  cat logs/<jobname>.out"
echo " View errors:  cat logs/<jobname>.err"
echo "============================================="
