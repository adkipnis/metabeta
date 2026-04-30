#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=[a100_40gb|a100_80gb|h100_80gb]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

EPOCHS=6000
ACCUM=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        --model_id) MODEL_ID="$2"; shift 2 ;;
        --valid_ds_type) VALID_DS_TYPE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --accum) ACCUM=1; shift ;;
        --latest) LOAD_LATEST=1; shift ;;
        --best) LOAD_BEST=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$TAG" ]] && { echo "Usage: $0 --data_id <size-family-ds_type> [--valid_ds_type <type>] [--epochs N] [--seed N] [--latest|--best]"; exit 1; }
[[ "$LOAD_LATEST" -eq 1 && "$LOAD_BEST" -eq 1 ]] && { echo "Error: --latest and --best are mutually exclusive"; exit 1; }

IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"
case $FAM_NAME in
    n) FAMILY=0 ;;
    b) FAMILY=1 ;;
    p) FAMILY=2 ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training

EXTRA_ARGS=()
if [[ "$ACCUM" -eq 1 ]]; then
    EXTRA_ARGS+=(--bs 16 --accum_steps 2)
fi
if [[ "$LOAD_LATEST" -eq 1 ]]; then
    EXTRA_ARGS+=(--load_latest)
fi
if [[ "$LOAD_BEST" -eq 1 ]]; then
    EXTRA_ARGS+=(--load_best)
fi

python train.py \
    --size "${SIZE}" \
    --model_id "${MODEL_ID:-${SIZE}}" \
    --family ${FAMILY} \
    --ds_type "${DS_TYPE}" \
    ${VALID_DS_TYPE:+--valid_ds_type ${VALID_DS_TYPE}} \
    -e ${EPOCHS} \
    ${SEED:+--seed ${SEED}} \
    "${EXTRA_ARGS[@]}" \
    --wandb \
    --device cuda \
    --no-plot
