#!/bin/bash
# t-proposal evidence (experiments/posthoc/evidence.py --t-only) on the rows of the existing
# evidence CSVs as a GPU array, one task per (family, size, shard); the CPU merge of
# evidence-array.sh then joins the t columns onto the CSVs and rewrites the summaries:
#   mkdir -p logs/evidence-t && JOB=$(sbatch --parsable scripts/evidence-t.sh)
#   sbatch --dependency=afterany:$JOB --array=0 scripts/evidence-array.sh --merge

#SBATCH --job-name=evidence-t
#SBATCH --output=logs/evidence-t/%A_%a.out
#SBATCH --error=logs/evidence-t/%A_%a.err
#SBATCH --array=0-47

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --exclude=supergpu19
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

FAMILIES=(normal bernoulli poisson)
SIZES=(small medium large huge)
N_SHARDS=4
TASK=${SLURM_ARRAY_TASK_ID:-0}
COMBO=$((TASK / N_SHARDS))

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta
export PYTHONPATH=$PWD

python experiments/posthoc/evidence.py --family "${FAMILIES[$((COMBO / ${#SIZES[@]}))]}" \
    --sizes "${SIZES[$((COMBO % ${#SIZES[@]}))]}" --n-datasets 512 --t-only --device cuda \
    --shard $((TASK % N_SHARDS)) --n-shards $N_SHARDS "$@"
