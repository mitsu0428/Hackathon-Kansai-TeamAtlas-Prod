#!/bin/bash
#SBATCH --job-name=baseline-update
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/baseline_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

echo "Starting baseline update job"
echo "Working dir: $(pwd)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

PYTHON=backend/.venv/bin/python

export DEVICE=cuda
export FAISS_USE_GPU=true

$PYTHON batch/jobs/run_update.py "$@"

echo "Baseline update complete"
