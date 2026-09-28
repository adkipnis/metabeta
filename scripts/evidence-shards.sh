#!/bin/bash
# Run experiments/posthoc/evidence.py as parallel shards on one (already allocated) node,
# then merge the shard CSVs and write the summary.
#
# usage: scripts/evidence-shards.sh FAMILY SIZE [N_SHARDS] [extra evidence.py args ...]
#   N_SHARDS defaults to half the allocated cores (2 torch threads per shard).
#   e.g.   scripts/evidence-shards.sh bernoulli small
#          scripts/evidence-shards.sh poisson huge 32 --n-datasets 256
# Logs: logs/evidence/{family}_{size}/shard{k}.log
set -euo pipefail

FAMILY=$1
SIZE=$2
CORES=${SLURM_CPUS_ON_NODE:-$(nproc)}
N=${3:-$((CORES / 2))}
shift $(($# < 3 ? $# : 3))
THREADS=$((CORES / N > 0 ? CORES / N : 1))

cd "$(dirname "$0")/.."
LOG=logs/evidence/${FAMILY}_${SIZE}
mkdir -p "$LOG"
ARGS=(--family "$FAMILY" --sizes "$SIZE" --n-datasets 512 --nested 128 "$@")

echo "[$(date +%T)] $FAMILY/$SIZE: $N shards x $THREADS threads, logs in $LOG"
for ((k = 0; k < N; k++)); do
    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
        uv run python experiments/posthoc/evidence.py "${ARGS[@]}" --shard "$k" --n-shards "$N" \
        >"$LOG/shard$k.log" 2>&1 &
done
wait
echo "[$(date +%T)] $FAMILY/$SIZE: shards done, merging"
uv run python experiments/posthoc/evidence.py "${ARGS[@]}" --summarize-only
