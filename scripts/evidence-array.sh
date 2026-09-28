#!/bin/bash

# GLMM evidence check (experiments/posthoc/evidence.py) as a SLURM array: one task per
# (family, size, shard), N_SHARDS shards per (family, size), then a dependent merge job.
#   JOB=$(sbatch --parsable scripts/evidence-array.sh)
#   sbatch --dependency=afterany:$JOB --array=0 scripts/evidence-array.sh --merge
# Extra arguments are passed on to evidence.py (e.g. --n-datasets 256); pass the same to --merge.

#SBATCH --job-name=evidence
#SBATCH --output=logs/evidence/%A_%a.out
#SBATCH --error=logs/evidence/%A_%a.err
#SBATCH --array=0-511

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

FAMILIES=(bernoulli poisson)
SIZES=(small medium large huge)
N_SHARDS=64

MERGE=0
if [[ "${1:-}" == "--merge" ]]; then
    MERGE=1
    shift
fi
EXTRA="$*"

TASK=${SLURM_ARRAY_TASK_ID:-0}
COMBO=$((TASK / N_SHARDS))
SHARD=$((TASK % N_SHARDS))
FAMILY=${FAMILIES[$((COMBO / ${#SIZES[@]}))]}
SIZE=${SIZES[$((COMBO % ${#SIZES[@]}))]}

SIF="$HOME/containers/python312.sif"
VENV="$HOME/metabeta/.venv-apptainer"
# outputs/{data,checkpoints} are symlinked to workspace storage; bind it so the links resolve
WORKSPACE="/lustre/groups/hcai/workspace/alexander.kipnis"
BASE="--n-datasets 512 --nested 128"

mkdir -p logs/evidence

if [[ $MERGE -eq 1 ]]; then
    RUN=""
    for f in "${FAMILIES[@]}"; do
        for s in "${SIZES[@]}"; do
            RUN+="python experiments/posthoc/evidence.py --family $f --sizes $s $BASE $EXTRA --summarize-only; "
        done
    done
else
    RUN="python experiments/posthoc/evidence.py --family $FAMILY --sizes $SIZE $BASE $EXTRA --shard $SHARD --n-shards $N_SHARDS"
fi

apptainer exec \
  --bind "$HOME:$HOME" \
  --bind "$WORKSPACE:$WORKSPACE" \
  "$SIF" \
  bash -lc "
    set -euo pipefail
    cd '$HOME/metabeta'
    source '$VENV/bin/activate'
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
    export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
    echo 'hostname:' \$(hostname) ' task: $TASK ($FAMILY $SIZE shard $SHARD/$N_SHARDS, merge=$MERGE)'
    $RUN
  "
