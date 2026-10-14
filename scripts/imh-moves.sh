#!/bin/bash
# Oracle comparison of the IMH under-coverage remedy (experiments/posthoc/imh_moves.py),
# one array task per (family, size) test set:
#   sbatch --array=0-11 scripts/imh-moves.sh [imh_moves.py args, e.g. --variants imh gate]

#SBATCH --job-name=imh-moves
#SBATCH --output=logs/imh-moves/%A_%a.out
#SBATCH --error=logs/imh-moves/%A_%a.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --exclude=supergpu19
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

FAMILIES=(normal bernoulli poisson)
SIZES=(small medium large huge)
TASK=${SLURM_ARRAY_TASK_ID:-0}
FAMILY=${FAMILIES[$((TASK / 4))]}
SIZE=${SIZES[$((TASK % 4))]}

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta
export PYTHONPATH=$PWD

python experiments/posthoc/imh_moves.py --device cuda --families "$FAMILY" --sizes "$SIZE" "$@"
