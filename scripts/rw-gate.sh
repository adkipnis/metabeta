#!/bin/bash
# Gate calibration of the IMH's random-walk moves (experiments/posthoc/rw_gate.py), one array
# task per source: 0-11 simulated test sets, 12-23 real-data benchmark (no huge Poisson set:
# that task exits at once), 24 prior grid.
#   sbatch --array=0-24 scripts/rw-gate.sh [rw_gate.py args]

#SBATCH --job-name=rw-gate
#SBATCH --output=logs/rw-gate/%A_%a.out
#SBATCH --error=logs/rw-gate/%A_%a.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --exclude=supergpu19
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

FAMILIES=(normal bernoulli poisson)
SIZES=(small medium large huge)
TASK=${SLURM_ARRAY_TASK_ID:-0}

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta
export PYTHONPATH=$PWD

if [ "$TASK" -ge 24 ]; then
    python experiments/posthoc/rw_gate.py --device cuda --sources grid "$@"
else
    SOURCE=$([ "$TASK" -ge 12 ] && echo real || echo oracle)
    I=$((TASK % 12))
    python experiments/posthoc/rw_gate.py --device cuda --sources "$SOURCE" \
        --families "${FAMILIES[$((I / 4))]}" --sizes "${SIZES[$((I % 4))]}" "$@"
fi
