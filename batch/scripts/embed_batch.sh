#!/bin/bash
#SBATCH --job-name=clap-embed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/embed_%j.log

set -euo pipefail

# Use SLURM_SUBMIT_DIR (where sbatch was called) as project root
cd "${SLURM_SUBMIT_DIR:-.}"

echo "Starting CLAP embedding job"
echo "Working dir: $(pwd)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Use venv python directly (no source activate needed)
PYTHON=backend/.venv/bin/python

# Set GPU device
export DEVICE=cuda
export FAISS_USE_GPU=true

$PYTHON batch/jobs/run_embed.py "$@"

echo "Embedding job complete"
