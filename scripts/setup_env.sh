#!/bin/bash
# =============================================================================
# setup_env.sh — One-time environment setup for cluster nodes
# =============================================================================
# Usage:
#   bash scripts/setup_env.sh
#
# This script creates a Python virtual environment, installs all dependencies,
# and downloads CIFAR-10 so that training jobs don't waste GPU time on setup.
#
# Run this ONCE from the login node before submitting any PBS jobs.
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

echo "============================================="
echo " SCP-M Environment Setup"
echo " Project: ${PROJECT_DIR}"
echo "============================================="

# --- 1. Load modules (adjust to your cluster) ---
# Uncomment and modify for your cluster's module system:
# module purge
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9
echo "[1/4] Loading modules..."
echo "  (Edit this script to match your cluster's 'module load' commands)"

# --- 2. Create virtual environment ---
echo "[2/4] Creating virtual environment at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists. Skipping."
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# --- 3. Install dependencies ---
echo "[3/4] Installing dependencies..."
pip install -r "${PROJECT_DIR}/requirements.txt" --quiet
echo "  Dependencies installed."

# Verify GPU is accessible
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
"

# --- 4. Pre-download CIFAR-10 ---
echo "[4/4] Pre-downloading CIFAR-10 dataset..."
python -c "
from torchvision import datasets
datasets.CIFAR10(root='${PROJECT_DIR}/data/cifar10', train=True, download=True)
datasets.CIFAR10(root='${PROJECT_DIR}/data/cifar10', train=False, download=True)
print('  CIFAR-10 downloaded successfully.')
"

echo ""
echo "============================================="
echo " Setup complete!"
echo " Activate with: source ${VENV_DIR}/bin/activate"
echo " Submit jobs with: bash scripts/submit_all.sh"
echo "============================================="
