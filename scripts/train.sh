#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=logs/train/train_%j.out
#SBATCH --error=logs/train/train_%j.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=[a100_20gb|a100_80gb|a100_40gb|h100_80gb]
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

EPOCHS=1000

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag) TAG="$2"; shift 2 ;;
        --valid_ds_type) VALID_DS_TYPE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$TAG" ]] && { echo "Usage: $0 --tag <size-family-ds_type> [--valid_ds_type <type>] [--epochs N] [--seed N]"; exit 1; }

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
python train.py \
    --size "${SIZE}" \
    --family ${FAMILY} \
    --ds_type "${DS_TYPE}" \
    ${VALID_DS_TYPE:+--valid_ds_type ${VALID_DS_TYPE}} \
    -e ${EPOCHS} \
    ${SEED:+--seed ${SEED}} \
    --wandb \
    --device cuda \
    --no-plot
