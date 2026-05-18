#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=logs/train/%j.out
#SBATCH --error=logs/train/%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=[a100_40gb|a100_80gb|h100_80gb]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

EPOCHS=8000
ACCUM=0
PERMUTE=0
VALID_DS_TYPE=sampled

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_id) TAG="$2"; shift 2 ;;
        --model_id) MODEL_ID="$2"; shift 2 ;;
        --valid_ds_type) VALID_DS_TYPE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --accum) ACCUM=1; shift ;;
        --permute) PERMUTE=1; shift ;;
        --latest) LOAD_LATEST=1; shift ;;
        --best) LOAD_BEST=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$TAG" ]] && { echo "Usage: $0 --data_id <size-family-ds_type> [--model_id <id>] [--valid_ds_type <type>] [--epochs N] [--seed N] [--no-permute] [--accum] [--latest|--best]"; exit 1; }
[[ "${LOAD_LATEST:-0}" -eq 1 && "${LOAD_BEST:-0}" -eq 1 ]] && { echo "Error: --latest and --best are mutually exclusive"; exit 1; }

# When run as a SLURM array job without an explicit --seed, derive seed from task ID.
if [[ -z "$SEED" && -n "$SLURM_ARRAY_TASK_ID" ]]; then
    SEED=$(( SLURM_ARRAY_TASK_ID + 1 ))
fi

IFS='-' read -r SIZE FAM_NAME DS_TYPE <<< "$TAG"
case $FAM_NAME in
    n) FAMILY=0; FAMILY_NAME=normal ;;
    b) FAMILY=1; FAMILY_NAME=bernoulli ;;
    p) FAMILY=2; FAMILY_NAME=poisson ;;
    *) echo "Unknown family letter: $FAM_NAME (use n, b, or p)"; exit 1 ;;
esac
WANDB_SUFFIX="${SIZE}-${FAMILY_NAME}"

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta/metabeta/training

EXTRA_ARGS=()
if [[ "$ACCUM" -eq 1 ]]; then
    EXTRA_ARGS+=(--bs 16 --accum_steps 2)
fi
if [[ "${LOAD_LATEST:-0}" -eq 1 ]]; then
    EXTRA_ARGS+=(--load_latest)
fi
if [[ "${LOAD_BEST:-0}" -eq 1 ]]; then
    EXTRA_ARGS+=(--load_best)
fi
if [[ -n "$WANDB_SUFFIX" ]]; then
    EXTRA_ARGS+=(--wandb_suffix "$WANDB_SUFFIX")
fi
if [[ "$PERMUTE" -eq 1 ]]; then
    EXTRA_ARGS+=(--permute --r_tag perm)
fi

python train.py \
    --size "${SIZE}" \
    --model_id "${MODEL_ID:-${SIZE}}" \
    --family ${FAMILY} \
    --ds_type "${DS_TYPE}" \
    ${VALID_DS_TYPE:+--valid_ds_type ${VALID_DS_TYPE}} \
    -e ${EPOCHS} \
    ${SEED:+--seed ${SEED}} \
    --cores 4 \
    "${EXTRA_ARGS[@]}" \
    --wandb \
    --device cuda \
    --no-plot
