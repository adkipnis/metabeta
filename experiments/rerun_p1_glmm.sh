#!/usr/bin/env bash
# Priority-1 reruns after the Laplace mode-search robustification (branch
# posthoc-laplace-upgrades; see experiments/posthoc/LAPLACE_UPGRADES.md).
#
# Scope: the four robustness experiments feeding the main-text table
# tables/robustness_worst.tex, Bernoulli and Poisson only (the fix does not touch
# the Normal path, so Gaussian rows are unchanged). Defaults are kept (prefix
# 'latest', n_samples 1000, cpu) so the setup matches the existing tables; the
# refined-MB rows come from the family preset (imhLaplace).
#
# Lines are independent; split them across nodes as convenient. Run from repo root.
#
# Afterwards: experiments/results/{likelihood_misspec,prior_misspec,ood_design,
# condition_number}_{b,p}.tex flow into the paper tables as usual, and
# tables/robustness_worst.tex needs its Bernoulli/Poisson rows re-assembled from
# the per-experiment worst-condition rows.

set -euo pipefail

# posterior_eval's caches are mtime-fresh, so the pre-fix imhLaplace/isLaplace
# artifacts (refined-sample npz + summary .pt) would be served as-is; purge them
# first. MB pools and NUTS summaries stay cached — only refinement is recomputed.
# The name filter only ever matches GLMM artifacts (Normal uses imhMarginal).
find metabeta/outputs/data -type f \( -name '*imhLaplace*' -o -name '*isLaplace*' \) -print -delete

uv run python experiments/evaluation/likelihood_misspec.py --family b
uv run python experiments/evaluation/likelihood_misspec.py --family p

uv run python experiments/evaluation/prior_misspec.py --family b
uv run python experiments/evaluation/prior_misspec.py --family p

uv run python experiments/evaluation/ood_design.py --family b
uv run python experiments/evaluation/ood_design.py --family p

uv run python experiments/evaluation/condition_number.py --family b
# no huge-p-real test set exists (too few real datasets at that size)
uv run python experiments/evaluation/condition_number.py --family p --sizes small medium large
