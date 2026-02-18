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

# =========================================================================
# 1. Load cluster modules
# =========================================================================
# IMPORTANT: Uncomment and adjust the lines below for YOUR cluster.
# Run 'module avail' on your login node to see what's available.
#
# Common examples:
#   module purge
#   module load python/3.10
#   module load cuda/12.1
#   module load cudnn/8.9
#
# If your cluster provides PyTorch as a module (check with 'module avail pytorch'),
# load it here and set CLUSTER_HAS_PYTORCH=true below:
#   module load pytorch/2.1
# =========================================================================
echo "[1/4] Loading modules..."
echo "  (Edit this script to add your cluster's 'module load' commands)"

# Set to 'true' if your cluster provides PyTorch via 'module load'
CLUSTER_HAS_PYTORCH=false

# =========================================================================
# 2. Create virtual environment
# =========================================================================
echo "[2/4] Creating virtual environment at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists. Skipping."
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# =========================================================================
# 3. Install dependencies
# =========================================================================
echo "[3/4] Installing dependencies..."

if [ "${CLUSTER_HAS_PYTORCH}" = true ]; then
    # If PyTorch is provided by the cluster module system, only install
    # the non-PyTorch dependencies to avoid version conflicts.
    echo "  Using cluster-provided PyTorch. Installing other dependencies..."
    pip install numpy>=1.24.0 matplotlib>=3.7.0 pyyaml>=6.0 \
        tqdm>=4.65.0 pillow>=9.5.0 scipy>=1.10.0 --quiet
else
    # Install PyTorch from the official PyTorch pip index.
    # This provides CUDA-enabled wheels that the default PyPI does not have.
    #
    # Adjust the --index-url for your CUDA version:
    #   CUDA 11.8: https://download.pytorch.org/whl/cu118
    #   CUDA 12.1: https://download.pytorch.org/whl/cu121
    #   CUDA 12.4: https://download.pytorch.org/whl/cu124
    #   CPU only:  https://download.pytorch.org/whl/cpu
    echo "  Installing PyTorch from official index (CUDA 12.1)..."
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121 --quiet

    echo "  Installing remaining dependencies..."
    pip install numpy>=1.24.0 matplotlib>=3.7.0 pyyaml>=6.0 \
        tqdm>=4.65.0 pillow>=9.5.0 scipy>=1.10.0 --quiet
fi

echo "  Dependencies installed."

# Verify installation
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('  WARNING: CUDA not available. Training will be slow on CPU.')
    print('  Check that CUDA modules are loaded and the correct --index-url is used.')
"

# =========================================================================
# 4. Pre-download CIFAR-10
# =========================================================================
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
