#!/bin/bash

#SBATCH --job-name=oracle-corr
#SBATCH --output=logs/oracle-corr/%j.out
#SBATCH --error=logs/oracle-corr/%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Correlation-parameter oracle evaluation (app:corr) for one checkpoint on one sampled test set.
# Reuses the MB sample / refinement caches of the oracle run when present.
#   sbatch scripts/oracle-corr.sh --checkpoint data=huge-n-mixed_model=large_seed=16 --data_id huge-n-sampled --plot

PREFIX=latest
PLOT=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CKPT="$2"; shift 2 ;;
        --data_id) DATA_ID="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --plot) PLOT=(--plot); shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${CKPT:-}" || -z "${DATA_ID:-}" ]] && {
    echo "Usage: $0 --checkpoint <name|path> --data_id <size-family-sampled> [--prefix best|latest] [--plot]"
    exit 1
}

# Accept either a bare checkpoint name (resolved under outputs/checkpoints) or a full path.
if [[ "$CKPT" == */* ]]; then
    CKPT_DIR="$CKPT"
else
    CKPT_DIR="metabeta/outputs/checkpoints/${CKPT}"
fi

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta

python experiments/evaluation/oracle_corr.py \
    --checkpoint "${CKPT_DIR}" --data_id "${DATA_ID}" --prefix "${PREFIX}" \
    --device cuda --verbosity 1 "${PLOT[@]}"
