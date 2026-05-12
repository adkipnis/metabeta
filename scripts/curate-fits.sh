#!/bin/bash

#SBATCH --job-name=curate
#SBATCH --output=logs/curate/%j.out
#SBATCH --error=logs/curate/%j.err

#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${TAG:-}" ]] && { echo "Usage: $0 --data_id <size-family-ds_type> --partition <valid|test>"; exit 1; }
[[ -z "${PARTITION:-}" ]] && { echo "Usage: $0 --data_id <size-family-ds_type> --partition <valid|test>"; exit 1; }

case "$PARTITION" in
    valid|test) ;;
    *) echo "Unknown partition: $PARTITION (use valid or test)"; exit 1 ;;
esac

set +u
source $HOME/.bashrc
set -u
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta

echo "=== check.py ==="
python metabeta/simulation/check.py --data_id "$TAG" --partition "$PARTITION"

echo "=== cache.py ==="
python metabeta/evaluation/cache.py --data_id "$TAG" --partition "$PARTITION"
