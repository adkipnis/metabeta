#!/bin/bash

#SBATCH --job-name=test-real
#SBATCH --output=logs/test-real/%j.out
#SBATCH --error=logs/test-real/%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

PREFIX=best
DATA_IDS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CKPT="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --data_ids) shift; while [[ $# -gt 0 && "$1" != --* ]]; do DATA_IDS+=("$1"); shift; done ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${CKPT:-}" ]] && { echo "Usage: $0 --checkpoint <name|path> [--prefix best|latest] [--data_ids <id> [<id> ...]]"; exit 1; }

# Accept either a bare checkpoint name (resolved under outputs/checkpoints) or a full path.
if [[ "$CKPT" == */* ]]; then
    CKPT_DIR="$CKPT"
else
    CKPT_DIR="metabeta/outputs/checkpoints/${CKPT}"
fi

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta

# Eval RNG seed is fixed here; the training seed lives only in the checkpoint name.
ARGS=(--checkpoint "${CKPT_DIR}" --prefix "${PREFIX}" --device cuda --verbosity 1)
if [[ ${#DATA_IDS[@]} -gt 0 ]]; then
    ARGS+=(--data_ids "${DATA_IDS[@]}")
fi

python experiments/evaluation/real_posterior.py "${ARGS[@]}"
