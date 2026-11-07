#!/bin/bash
# MB on the whole real datasets beyond the routing limits (experiments/simulation/full_real_sets.py),
# with the checkpoints of the paper's real-data tables. A CPU node, since the padded tensors of
# muscatine and Chem97 outgrow a GPU at 4000 draws. Partition and QOS go on the command line:
#   sbatch -p cpu_p -q cpu_normal -c 16 --mem=300G -t 08:00:00 scripts/full-real.sh diagnostics
#   sbatch -p cpu_p -q cpu_normal -c 16 --mem=300G -t 12:00:00 scripts/full-real.sh compare
#   sbatch -p cpu_p -q cpu_normal -c 16 --mem=64G -t 04:00:00 scripts/full-real.sh subsamples
#   sbatch -p cpu_p -q cpu_normal -c 16 --mem=240G -t 08:00:00 scripts/full-real.sh within
#   sbatch -p cpu_p -q cpu_normal -c 4 --mem=16G -t 00:30:00 scripts/full-real.sh blocks
# compare needs the reintegrated fits (metabeta/simulation/check.py --partition test); blocks
# needs the per-dataset output of compare and within. An optional second argument names one
# whole set (e.g. small-p-full) to run diagnostics, compare or within on that set only.
# The code of METABETA_ROOT runs (default ~/metabeta), ahead of the venv's editable install.

#SBATCH --job-name=full-real
#SBATCH --output=logs/full-real/%j.out
#SBATCH --error=logs/full-real/%j.err
#SBATCH --nodes=1

set -euo pipefail
STAGE="$1"
ONLY="${2:-}"
METABETA_ROOT="${METABETA_ROOT:-$HOME/metabeta}"
cd "$METABETA_ROOT"
source "$HOME/metabeta/.venv/bin/activate"
export PYTHONPATH="$METABETA_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python -c 'import metabeta; print("metabeta:", metabeta.__file__)'

CK=metabeta/outputs/checkpoints
OUT=experiments/results/full-real
if [ "$STAGE" = subsamples ]; then
    # the same diagnostics on the subsamples of InstEval and Chem97 in the real-world test sets
    for spec in large-n-real:insteval__grp_group:data=large-n-mixed_model=large_seed=9 \
                small-p-real:chem97__grp_group:data=small-p-mixed_model=large_seed=4; do
        IFS=: read -r id source ckpt <<< "$spec"
        python experiments/evaluation/full_real_diagnostics.py --data_id "$id" --source "$source" \
            --checkpoint "$CK/$ckpt" --n_samples 4000 --out "$OUT/$id-$source-diagnostics-s4000.jsonl"
    done
    exit 0
fi
if [ "$STAGE" = blocks ]; then
    python experiments/evaluation/full_real_blocks.py --results "$OUT" "$OUT/within"
    exit 0
fi
for pair in large-n-full:data=large-n-mixed_model=large_seed=9 \
            medium-b-full:data=medium-b-mixed_model=large_seed=3 \
            small-p-full:data=small-p-mixed_model=large_seed=4; do
    id="${pair%%:*}"; ckpt="$CK/${pair#*:}"
    if [ -n "$ONLY" ] && [ "$id" != "$ONLY" ]; then continue; fi
    case "$STAGE" in
        diagnostics)
            python experiments/evaluation/full_real_diagnostics.py --data_id "$id" \
                --checkpoint "$ckpt" --n_samples 4000 --out "$OUT/$id-diagnostics-s4000.npz" ;;
        compare)
            python experiments/evaluation/real_posterior.py --checkpoint "$ckpt" --prefix latest \
                --n_samples 4000 --batch_size 1 --no-latency --data_ids "$id" --outdir "$OUT" ;;
        within)
            # the within-budget real-world test set of the same family and size regime, from the
            # caches behind the paper's tables; its own outdir, since table names key on the
            # checkpoint only
            python experiments/evaluation/real_posterior.py --checkpoint "$ckpt" --prefix latest \
                --n_samples 4000 --batch_size 1 --no-latency --data_ids "${id%-full}-real" \
                --outdir "$OUT/within" ;;
        *) echo "Unknown stage: $STAGE (use diagnostics, compare, within, subsamples or blocks)" >&2; exit 1 ;;
    esac
done
