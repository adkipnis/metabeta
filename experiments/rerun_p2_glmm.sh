#!/usr/bin/env bash
# Priority-2 reruns after the Laplace mode-search robustification (branch
# posthoc-laplace-upgrades; see experiments/posthoc/LAPLACE_UPGRADES.md). Appendix
# artifacts with refined-MB rows on Bernoulli/Poisson; Normal is untouched by the fix.
#
# Ordered by how much the fix is expected to move things. Lines are independent —
# split across nodes as convenient (the oracle/real blocks are the heavy ones). Run
# from repo root on a CPU node. Note for resumes: while METABETA_REFRESH_METHODS is
# set, already-refreshed dirs are recomputed again; comment out completed lines.

set -euo pipefail
export METABETA_REFRESH_METHODS="imhLaplace,isLaplace"   # bypass pre-fix refinement caches

CKPT=metabeta/outputs/checkpoints
# checkpoint dirs per (family, size): scripts/build_ckpt.py BEST_SEEDS
ckpt() { echo "$CKPT/data=$2-$1-mixed_model=large_seed=$3"; }


# 1) oracle benchmarks (appendix oracle_{bernoulli,poisson}.tex; largest expected shift, e.g.
#    Poisson-large refined sigma_rfx ECE -0.070 -> -0.051)
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt b small 6)"   --data_id small-b-sampled
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt b medium 3)"  --data_id medium-b-sampled
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt b large 4)"   --data_id large-b-sampled
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt b huge 8)"    --data_id huge-b-sampled
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt p small 4)"   --data_id small-p-sampled
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt p medium 11)" --data_id medium-p-sampled
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt p large 6)"   --data_id large-p-sampled
uv run python experiments/evaluation/oracle_posterior.py --checkpoint "$(ckpt p huge 9)"    --data_id huge-p-sampled

# 2) real-world NUTS agreement (appendix real_{bernoulli,poisson}.tex); no huge-p-real set exists
uv run python experiments/evaluation/real_posterior.py --checkpoint "$(ckpt b small 6)"   --data_ids small-b-real
uv run python experiments/evaluation/real_posterior.py --checkpoint "$(ckpt b medium 3)"  --data_ids medium-b-real
uv run python experiments/evaluation/real_posterior.py --checkpoint "$(ckpt b large 4)"   --data_ids large-b-real
uv run python experiments/evaluation/real_posterior.py --checkpoint "$(ckpt b huge 8)"    --data_ids huge-b-real
uv run python experiments/evaluation/real_posterior.py --checkpoint "$(ckpt p small 4)"   --data_ids small-p-real
uv run python experiments/evaluation/real_posterior.py --checkpoint "$(ckpt p medium 11)" --data_ids medium-p-real
uv run python experiments/evaluation/real_posterior.py --checkpoint "$(ckpt p large 6)"   --data_ids large-p-real

# 3) agreement figures (pool small/medium x n/b/p real by default; Normal is served from cache)
( cd experiments/evaluation && uv run python agreement_scatter.py && uv run python agreement_marginals.py )

# 4) data-poverty appendix tables
uv run python experiments/evaluation/data_poverty.py --family b
uv run python experiments/evaluation/data_poverty.py --family p

# 5) posthoc ablation appendix tables (ablation.py has its own summary cache -> --refresh-summaries;
#    the refined rows are hand-ported from metabeta/outputs/results/ablation/{family}_{size}.md)
uv run python experiments/posthoc/ablation.py --sizes small medium large huge --families bernoulli poisson \
    --split test --refresh-summaries

# 6) MB-NUTS spot check: warm-start seeds come from imhLaplace draws; the escalation fallback makes
#    movement unlikely, so verify on one regime before deciding on a full --wn-refit rerun
uv run python experiments/posthoc/ablation.py --sizes large --families poisson --split test --n-datasets 128 \
    --only imhLaplace coldNuts warmNuts --include-warmnuts --wn-refit --refresh-summaries
