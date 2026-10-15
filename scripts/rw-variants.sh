#!/bin/bash
# Paired RW settings (experiments/posthoc/rw_variants.py), one array task per set:
# 0-5 simulated large/huge (normal, bernoulli, poisson), 6-12 real-data benchmark.
#   sbatch --array=0-12 scripts/rw-variants.sh [rw_variants.py args]

#SBATCH --job-name=rw-variants
#SBATCH --output=logs/rw-gate/variants_%A_%a.out
#SBATCH --error=logs/rw-gate/variants_%A_%a.err

#SBATCH --partition gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --exclude=supergpu19
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00

SETS=(oracle:normal_large oracle:normal_huge oracle:bernoulli_large oracle:bernoulli_huge
      oracle:poisson_large oracle:poisson_huge real:normal_medium real:normal_large
      real:bernoulli_medium real:bernoulli_large real:bernoulli_huge real:poisson_medium
      real:poisson_large)
SPEC=${SETS[${SLURM_ARRAY_TASK_ID:-0}]}

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate
cd $HOME/metabeta
export PYTHONPATH=$PWD

python experiments/posthoc/rw_variants.py --device cuda --sources "${SPEC%%:*}" --sets "${SPEC##*:}" "$@"
