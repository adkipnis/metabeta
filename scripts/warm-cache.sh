#!/bin/bash
# Tar the PyTensor compile directory of scripts/warm_cache.py on small-{family}-sampled to
# ~/pytensor_cache_{family}.tar, for fit-ref.sh --warm:
#   sbatch scripts/warm-cache.sh --family n

#SBATCH --job-name=warm-cache
#SBATCH --output=logs/warm/%j.out
#SBATCH --error=logs/warm/%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_preemptible
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

FAMILY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --family) FAMILY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ "$FAMILY" =~ ^[nbp]$ ]] || { echo "Usage: $0 --family <n|b|p>" >&2; exit 1; }

OUT="$HOME/pytensor_cache_${FAMILY}.tar"

source "$HOME/metabeta/scripts/pymc-run.sh"
runPymc python scripts/warm_cache.py --data_id "small-$FAMILY-sampled"
tar cf "$OUT" -C "$PYTENSOR_DIR" .
echo "wrote $OUT ($(du -sh "$OUT" | cut -f1))"
