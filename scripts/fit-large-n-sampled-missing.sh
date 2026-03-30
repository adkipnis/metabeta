#!/bin/bash

#SBATCH --job-name=fit-large-n-sampled-missing
#SBATCH --output=logs/fit/large-n-sampled-missing_%A_%a.out
#SBATCH --error=logs/fit/large-n-sampled-missing_%A_%a.err
#SBATCH --array=0-83

#SBATCH --partition cpu_p
#SBATCH --qos cpu_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source $HOME/.bashrc
source $HOME/metabeta/.venv/bin/activate

JOB_TMPDIR="$HOME/tmp/pytensor_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export PYTENSOR_FLAGS="base_compiledir=$JOB_TMPDIR"

# Missing indices from check output:
# 001, 002, 004, 006, 007, 008, 011, 013, 014, 015,
# 017, 021, 022, 024, 026, 027, 028, 029, 030, 032,
# 033, 035, 038, 039, 040, 042, 043, 047, 048, 049,
# 051, 052, 054, 058, 059, 060, 061, 062, 063, 064,
# 065, 066, 067, 068, 070, 074, 075, 076, 077, 078,
# 080, 081, 083, 084, 085, 086, 087, 088, 092, 093,
# 094, 095, 096, 097, 098, 099, 102, 103, 104, 105,
# 106, 107, 108, 109, 113, 114, 115, 116, 117, 118,
# 119, 120, 122, 124
MISSING_IDXS=(1 2 4 6 7 8 11 13 14 15 17 21 22 24 26 27 28 29 30 32 33 35 38 39 40 42 43 47 48 49 51 52 54 58 59 60 61 62 63 64 65 66 67 68 70 74 75 76 77 78 80 81 83 84 85 86 87 88 92 93 94 95 96 97 98 99 102 103 104 105 106 107 108 109 113 114 115 116 117 118 119 120 122 124)
IDX="${MISSING_IDXS[$SLURM_ARRAY_TASK_ID]}"

cd "$HOME/metabeta/metabeta/simulation"
python fit.py --d_tag large-n-sampled --idx "${IDX}" --method nuts --loop
python fit.py --d_tag large-n-sampled --idx "${IDX}" --method advi

rm -rf "$JOB_TMPDIR"
