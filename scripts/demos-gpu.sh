#!/bin/bash
# Run a demo notebook through nbmake on one GPU node, without writing its outputs back. Submit from
# the root of the checkout to test (a separate worktree, so ~/metabeta and its jobs stay untouched);
# partition, QOS and GPU constraint go on the command line:
#   mkdir -p logs/demos-gpu
#   sbatch --partition gpu_p --qos gpu_normal --constraint h100_80gb \
#       scripts/demos-gpu.sh demos/04-many-datasets.ipynb

#SBATCH --job-name=demos-gpu
#SBATCH --output=logs/demos-gpu/%j.out
#SBATCH --error=logs/demos-gpu/%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source "$HOME/.bashrc"
set -euo pipefail

NOTEBOOK=${1:?usage: sbatch [--partition …] [--qos …] [--constraint …] scripts/demos-gpu.sh <notebook>}

# sbatch runs a spooled copy of this script, so the checkout is the submit directory
cd "$SLURM_SUBMIT_DIR"
uv sync
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run pytest --nbmake "$NOTEBOOK"
