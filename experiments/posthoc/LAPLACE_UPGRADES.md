# Laplace / IMH posthoc upgrades — record of branch `posthoc-laplace-upgrades` (2026-09)

## What the branch delivers

1. **Robustified Laplace mode search** (`posthoc/laplace_glmm.py::laplaceRfxModes`): damped
   Newton with a per-entry backtracking line search, an adaptive iteration budget driven by
   the Newton decrement, and a 1-nat pinning guard for samples whose mode search does not
   resolve. Fixes warm-start-dependent IMH weights (a correctness bug, see below).
2. **IMH is the API default** (`Api.sample(refine=True)`): the flow pool is over-drawn by
   the burn-in so exactly `n_samples` draws come back; `refine='is'` keeps the SNIS/PSIS
   path, `refine=False` the raw flow. Refinement applies from `MIN_SAFEGUARD_SAMPLES` upward.
3. **Acceptance-based pool-size suggestion** (`metropolis.suggestPoolSize`, in the IMH
   diagnostics and `RouterResult.safeguards`), calibrated by a pool-size sweep; mean
   acceptance < 0.1 warns with the concrete re-run size.
4. **Split-wide padding trimmed per chunk** (`dataloader.trimBatchPadding`,
   `Proposal.resizeGroups`) in `posterior_eval.refineProposal`, `getSummary`, and the API's
   chunked refinement — 3–4× fewer cells on the test splits, bit-identical results.
5. **`METABETA_REFRESH_METHODS`** env switch in `posterior_eval` to bypass per-method caches
   after a code change (cache freshness is mtime-based and blind to code).
6. Diagnostics: `newton_stability.py` (warm-start dependence of Laplace weights),
   `laplace_budget_timing.py` (cost profile of the weight pass / full refinement on a cached
   pool, `--full` for the exact oracle path).

## Why: the analysis that started it

Cross-referencing the posthoc ablations (`metabeta/outputs/results/ablation/`):
- The huge-regime under-dispersion of `imhLaplace` (FFX ECE ≈ −0.05) is **not** Laplace
  bias: `imhMarginal` on Normal-huge, with an exact target, shows the same signature at the
  same acceptance. It is the finite proposal pool.
- The Laplace-attributable σ_rfx shift was clearest on Poisson-large (`isLaplace` −0.046 vs
  raw −0.027; exact-target control −0.000) — and turned out to be mostly Newton instability.
- The conditional p(α_j | θ, y_j) is strictly log-concave for Bernoulli/Poisson canonical
  links: no fat tails, only skew. Upgrades aimed at skew (SIR) and integrated-likelihood bias
  (AGQ) were therefore tested — and retired.

## Newton stability: the bug and the fix

`newton_stability.py` (Poisson-large, 32 datasets, s=512): the median warm-start dependence
of a weight was 0.0000 nats, but on the huge-count dataset (y_max = 13 689) two warm starts
disagreed by up to 7e4 nats **on samples at the pool's top weight**, and even two 30-iteration
runs by 1.25e5 — full-step Newton oscillates on the clipped Poisson objective. After the fix:
budget independence 0.008 nats, MH decision flip rate 0.27 % → 0.01 %, remaining asymmetry
conservative only (a legit sample may be pinned, never a spurious one kept).

Cluster effect (512 test datasets): Poisson-large σ_rfx ECE `isLaplace` −0.046 → −0.031,
`imhLaplace` −0.070 → −0.051; Bernoulli-huge max PSIS k̂ 15.4 → 2.6; LOO-NLL unchanged at
NUTS level everywhere. End-to-end on the flagged dataset (same pool, vs NUTS): pre-fix output
was *not* visibly wrong (max |mean − NUTS|/sd 0.16 → 0.14, mixing 347 → 394 unique states);
all other datasets bit-identical. Kept as a tail-risk correctness fix; cost 1.7× per Newton
pass, more than offset by the padding trim.

## Tried and retired: AGQ weights, SIR redraw

Implemented (commits a32922f8 / 169cd495 hold the code) and cluster-ablated at 512 datasets:
AGQ (nAGQ=9, q ≤ 2) ≡ Laplace to ~3 decimals post-fix; SIR redraw (K=16, defensive
Laplace+prior mixture) small never-worse gains (Bernoulli-huge σ_rfx EACE 0.034 vs 0.043) at
~1.7× cost. Both removed again. Backlog: pseudo-marginal (GIMH) weights, INLA-style skew
corrections — only worth revisiting if flow proposal quality drops.

## Finite-pool sweep and the sizing rule

Bernoulli-huge, 512 datasets, `imhLaplace`:

| s | FFX ECE | σ_rfx ECE | Corr R | acceptance | raw FFX ECE |
|---|---|---|---|---|---|
| 1000 | −0.048 | −0.021 | 0.132 | 0.175 | −0.000 |
| 2000 | −0.031 | −0.017 | 0.245 | 0.168 | −0.000 |
| 4000 | −0.022 | −0.008 | 0.262 | 0.168 | +0.001 |

FFX ECE ∝ (ā·s)^−0.6 (s=4000 predicted −0.020 out-of-sample, observed −0.022); the raw
control is s-invariant and acceptance pool-size-independent. Small-regime calibration is
reached at ā·s ≈ 700 accepted draws → `suggestPoolSize` = ⌈700/ā⌉, multiples of 500,
clamped to [1000, 16000]. Benchmarks keep the fixed 1000-proposal pool (one variable per
comparison, paired caches); the suggestion is a reported diagnostic.

## Padding trim

`refineProposal` sliced chunks out of the fully collated batch, so every chunk carried the
split-wide (m, n) padding: 1.6× the cells at 32 datasets, 3.4× at 512. Trimming per chunk:
weights bit-identical, weight pass 1.8× faster, full imhLaplace refinement 0.80 → 0.46 s/ds
locally; on the cluster's GPU node the oracle refined time went 6.44 s/ds (branch, pre-trim)
→ 2.50 (branch) vs 2.78 (`main`). Node facts that mattered: 4 allotted cores, torch pinned to
4 (no oversubscription); a server core is ~4× slower than an M-series core on this workload.

## Paper impact and rerun verdict

Priority-1 reruns (robustness experiments behind `tables/robustness_worst.tex`, Bernoulli
and Poisson, cluster, caches bypassed): third-decimal shifts; four rounded cells moved by
0.01 toward NUTS (Cauchy predictors ±, collinearity 0.95 → 0.96 / 0.92 → 0.93). Targeted
oracle checks on the most sensitive regimes (same pools): Poisson-large refined row
identical at 2 dp, Bernoulli-huge one cell 0.01. **No further oracle / real / data-poverty /
ablation-table reruns are needed for the paper's accuracy claims.** Ported to the paper:
robustness tables (current renderer layout, cov90 dropped), IMH text (pool suggestion,
released vs benchmark pool geometry, mode search, acceptance gate) under `TODO(alex)`
markers. The oracle tables' time column is GPU-era and unrelated to this branch; refresh
with a dedicated timing run if quoted.

## Reproduction

```bash
# Newton warm-start diagnostic
uv run python experiments/posthoc/newton_stability.py --families poisson --sizes large --n-datasets 32 --n-samples 512
# cost profile on a cached pool (weight pass; --full = exact oracle refinement path)
uv run python experiments/posthoc/laplace_budget_timing.py --pool <test-*.mb.*.npz> --data-dir <data dir> --likelihood-family 2 --n-datasets 64 [--full --batch-size 8]
# posthoc ablation with fresh summaries (pool-size sweep: vary --n-samples; files get an _s{n} tag)
uv run python experiments/posthoc/ablation.py --sizes huge --families bernoulli --split test --only raw isLaplace imhLaplace --n-samples 4000 --refresh-summaries --device cuda
# evaluation scripts after a refinement code change (bypass stale caches)
METABETA_REFRESH_METHODS="imhLaplace,isLaplace" uv run python experiments/evaluation/oracle_posterior.py --checkpoint <ckpt dir> --data_id large-p-sampled --device cuda
```
