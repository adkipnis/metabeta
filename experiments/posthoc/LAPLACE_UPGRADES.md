# Laplace posthoc upgrades: analysis, experiments, and status

Branch: `posthoc-laplace-upgrades`. Working doc for testing three refinements of the
GLMM Laplace machinery in `metabeta/posthoc/laplace_glmm.py` / `metabeta/posthoc/metropolis.py`.

## Why (analysis summary, 2026-09-12)

Cross-referencing the posthoc ablations in `metabeta/outputs/results/ablation/`:

1. **The huge-regime under-dispersion of imhLaplace is NOT Laplace bias.** On Normal-huge,
   `imhMarginal` targets the *exact* marginal (Normal-Normal conjugacy) and still shows the
   same signature as the GLMM huge runs: FFX ECE −0.049, σ_rfx ECE −0.045 at acceptance
   0.151 (vs imhLaplace: −0.047/−0.063 FFX at acceptance 0.171/0.096 on b/p-huge). This is
   the finite-pool IMH effect — the chain can only concentrate the flow's draws, never
   create tail mass the flow lacks. No conditional-posterior upgrade can fix it.
2. **The genuine Laplace-attributable bias is modest and clearest on Poisson-large** (SNIS
   path, no chain confound): `isLaplace` σ_rfx ECE −0.046 vs raw −0.027, while the
   exact-target Normal-large control (`isMarginal`) sits at −0.000. Bernoulli-large shows
   almost nothing (−0.000 → −0.007). Predictively, imhLaplace already matches cold NUTS
   LOO-NLL to 3 decimals at every size.
3. **The conditional p(α_j | θ, y_j) is log-concave** for Bernoulli/Poisson canonical links
   (concave log-lik in η, η linear in α, Gaussian prior) → sub-Gaussian tails, *no fat
   tails*. The Gaussian N(b*, H⁻¹) redraw's real defect is **skew** (all-0/all-1 binary
   groups, low-count Poisson). Fat tails live in the global posterior — the flow's job.
4. **Subtlety**: `laplaceRfxModes` warm-starts Newton at the proposal's paired flow rfx
   draws, so the "deterministic" Laplace weight is a stochastic function of that draw if
   Newton under-converges within `n_newton=5`. Harmless weight noise in SNIS; in IMH it
   wobbles the pseudo-target without pseudo-marginal (GIMH) unbiasedness backing it.

Backlogged (deliberately not tested now): GIMH / pseudo-marginal weights (exact target but
weight noise sums over m groups — hurts exactly where acceptance is already low), INLA-style
deterministic skew corrections (more code, still biased; only worth it if AGQ+SIR leave
measurable residue), local-flow-as-SIR-proposal (too much compute for a practical
refinement step).

## Experiments

Common design: **always run an equal-sized fresh reference (raw, isLaplace, imhLaplace)
first**, on the identical subset and seed, and compare paired summaries. Primary test bed
is Poisson-large (biggest signal per point 2 above).

**Proof-of-concept scope (local MacBook, CPU, 24 GB RAM): 32 datasets, 512 flow samples.**
Promising results → extend on the cluster (commands below; 128–512 datasets, s=1000).

### (1) Newton warm-start stochasticity — diagnostic only

`experiments/posthoc/newton_stability.py`: same proposal pool; Laplace log-weights under
init ∈ {flow rfx A, independent flow rfx B, zeros} × n_newton ∈ {5, 10, 20}; reference =
zero-init n_newton=30. Reports |Δlog w| between inits (stochasticity), vs reference (bias),
and the fraction of common-random-number MH accept decisions that flip A→B.

Decision rule: median |Δlog w| ≲ 0.1 and flip rate < 1 % → negligible, document; else add a
step-norm convergence check to `laplaceRfxModes` (cap ~10 iters) and re-compare.

Cost: one-off diagnostic. If the convergence check is adopted: imhLaplace ≈ 1.1–1.2×.

### (2) AGQ weights for q ≤ 2 (`isAGQ`, `imhAGQ`)

`logMarginalLikelihoodAGQ` in `laplace_glmm.py`: reuse the Newton modes/Hessians, evaluate
adaptive Gauss–Hermite nodes α_k = b* + √2·L_H⁻ᵀ z_k, logsumexp with the ‖z‖² correction —
the multivariate generalization of the identity already used in
`refineBernoulliNagqSrfx` (`metabeta/analytical/glmm/bernoulli.py`). Nodes/dim: 9 (q=1),
7 (q=2); q > 2 falls back to plain Laplace (masked mix within a batch). Redraw unchanged.
Targets the σ_rfx weight bias (point 2). Watch PSIS fallback — may *drop*, since the flow
was trained on exact posteriors and the AGQ target is closer to exact than Laplace.

Est. slowdown vs imhLaplace (~0.4 s/ds at s=1000 on the ablation host): node evals are bare
likelihood passes, ~3–5× cheaper than Newton iterations → q=1: ~1.2–1.3×; q=2 (49 nodes):
~2–3×; mixed-q batch: ~1.5–2× overall.

### (3) SIR redraw (`isSIR`, `imhSIR`)

`sampleRfxSIR` in `laplace_glmm.py`: K=16 candidates per (dataset, group, sample) from the
defensive mixture 0.9·N(b*, H⁻¹) + 0.1·N(0, Σ_rfx); log-weight = exact conditional −
mixture density; Gumbel-max resample. Weights provably bounded (bounded GLM likelihood +
prior mixture component). Targets the redraw skew (point 3). Composes with (2).

Est. slowdown: +K bare likelihood evals on the redraw pass → K=16: ~1.4–1.6×; K=32: ~1.8–2×.
Combined (2)+(3): ~2–3.5× imhLaplace — still ~100× under NUTS.

## Commands

Local PoC (from repo root; `--n-samples 512` also sets the IMH pool to 4×128):

```bash
# Phase 0 — reference
uv run python experiments/posthoc/ablation.py --sizes large --families poisson \
    --split test --n-datasets 32 --n-samples 512 --only raw isLaplace imhLaplace
# Phase 1 — Newton stability diagnostic
uv run python experiments/posthoc/newton_stability.py --size large --family poisson \
    --n-datasets 32 --n-samples 512
# Phase 2 / 3 — candidates (after implementation)
uv run python experiments/posthoc/ablation.py --sizes large --families poisson \
    --split test --n-datasets 32 --n-samples 512 --only isAGQ imhAGQ isSIR imhSIR
```

Cluster scale-up (identical, bigger): `--n-datasets 512 --n-samples 1000` (drop
`--n-datasets` for the full split), plus a Bernoulli-small spot check for SIR
(`--sizes small --families bernoulli`, tiny groups = worst skew).

## Metrics & decision rules

σ_rfx ECE (primary for AGQ; expect the −0.046 excess to move toward raw's −0.027),
RFX R/ECE + joint ECE + LOO-NLL (primary for SIR — LOO is computed from conditional draws),
PSIS k̄/fallback, IMH acceptance, time/ds. Hard correctness gates in
`tests/utils/test_laplace_glmm.py`: AGQ must equal the exact Normal marginal; SIR must
match the exact Normal conditional (`sampleRfxConditionalNormal`).

## Results (PoC: Poisson-large, test split, 32 datasets, s=512, MacBook CPU)

### Phase 0 — pre-fix reference (`poisson_large_raw-isLaplace-imhLaplace.md`)

|            | σ_rfx ECE | FFX ECE | LOO-NLL | notes |
|------------|-----------|---------|---------|-------|
| raw        | −0.119    | −0.008  | 1.641   | |
| isLaplace  | −0.148    | −0.030  | 1.432   | k̄=0.72, fallback 34% |
| imhLaplace | −0.222    | −0.088  | 1.396   | acceptance 0.213, 0.1 s/ds |

Reproduces the full-run pattern at PoC scale; the smaller pool (512 vs 1000)
amplifies the finite-pool IMH under-dispersion, as expected.

### Phase 1 — Newton warm-start stochasticity (`newton_stability_poisson_large.md`)

Pre-fix: typical case negligible (median |Δlog w| between two warm starts ≈ 0.0000
vs pool std 7.9; decision flip rate 0.27 %) — **but** a rare catastrophic tail:
init-dependent weight swings of 2e4–7e4 nats concentrated on specific datasets, ON
SAMPLES AT/NEAR THE POOL MAX WEIGHT (ds 6: sample 241 is −16 nats from pool max
under init A, −70 580 under init B; sample 245 the reverse). Even two "converged"
runs disagreed by up to 1.25e5 nats (A30 vs Z30) — full-step Newton oscillates on
extreme Poisson proposals; the ±20 clamp masked it. These are init-dependent
absorbing states in IMH and weight-poisoners in SNIS (likely feeding the PSIS
fallback rate).

**Fixes adopted** (in `laplaceRfxModes` / the marginal functions):
1. per-entry backtracking line search (objective-based step halving);
2. adaptive iteration budget — the per-iteration Newton decrement λ²/2 is free from
   (score, delta), so the loop extends past n_newton (cap +15) only while some
   entry is unresolved; clean datasets pay nothing;
3. a 1-nat pinning guard — samples whose summed decrement stays > 1 nat get
   ll = −1e10 (they cannot become absorbing states / SNIS poison).

Root cause of the hard cases: dataset 6 is a huge-count Poisson dataset
(y_max = 13 689 vs ≤ 2 437 elsewhere, q=3) — μ ~ 1e4 makes the conditional
razor-sharp and the eta-clip plateaus the objective. Post-fix diagnostic:
within-init budget independence 0.008 nats max (was 2e3–5e3); MH decision flip
rates 0.01 % / 0.00 % (was 0.27 % / 0.14 %); remaining init-asymmetric pinning is
conservative only (a legit sample may be dropped, never a spurious one kept).
Regression test: `test_newton_backtracking_init_independence`. Cost: weight pass
~2.3× pre-fix on this 32-ds subset (48.8 s → 114.6 s for 8 passes), driven by the
backtracking objective evaluations.

### Phases 2+3 — PoC comparison (post-fix, all conditions on one shared pool)

`poisson_large_raw-isLaplace-imhLaplace-isAGQ-imhAGQ-isSIR-imhSIR.md`:

|            | σ_rfx ECE | FFX ECE | Corr R | LOO-NLL | time/ds |
|------------|-----------|---------|--------|---------|---------|
| raw        | −0.119    | −0.008  | 0.501  | 1.641   | — |
| isLaplace  | −0.148    | −0.030  | 0.501  | 1.411   | — |
| isAGQ      | −0.148    | −0.030  | 0.501  | 1.413   | — |
| isSIR      | −0.148    | −0.030  | 0.501  | 1.412   | — |
| imhLaplace | −0.157    | −0.104  | 0.321  | 1.397   | 0.3 s |
| imhAGQ     | −0.154    | −0.098  | 0.505  | 1.397   | 0.4 s |
| imhSIR     | −0.198    | −0.099  | 0.522  | 1.397   | 0.4 s |

Reading (32 datasets — direction only, not significance):
- **The Newton robustness fix is the substantive change.** Post-fix imhLaplace
  σ_rfx ECE improved −0.222 → −0.157 vs the pre-fix reference (same subset/seed),
  consistent with removing init-dependent absorbing states. Cost: IMH 0.1 → 0.3 s/ds
  (~2.5–3×, still ~300× under NUTS). On real flow proposals only ~3/256 samples of
  the pathological dataset get pinned.
- **AGQ and SIR are correct but sub-noise at this scale.** On real proposals the
  AGQ−Laplace weight correction is O(0.01–0.2) nats/dataset (verified directly;
  sign dataset-specific) — invisible in 32-ds metrics. The unit-test gates are the
  correctness evidence (AGQ exact for Normal, beats Laplace vs brute-force
  quadrature on tiny groups; SIR matches the exact Normal conditional and the
  grid-integrated skewed Bernoulli conditional mean). The σ_rfx-bias effect
  (≈0.02 ECE at 512 ds) needs the cluster scale to resolve.
- Corr R 0.321 for imhLaplace is chain-level noise on this subset (imhAGQ/imhSIR,
  same weights ±0.2 nats, sit at 0.505/0.522).
- Measured slowdowns vs imhLaplace: **imhAGQ 1.4×, imhSIR 1.4×** (12.7 s vs 9.1 s
  per 32 ds) — at or below the pre-run estimates.

## Cluster scale-up commands

Code changed → pass `--refresh-summaries` everywhere. From repo root:

```bash
# main comparison, full test split, s=1000 (Poisson-large: clearest weight-bias signal)
uv run python experiments/posthoc/ablation.py --sizes large --families poisson --split test \
    --only raw isLaplace imhLaplace isAGQ imhAGQ isSIR imhSIR --refresh-summaries
# worst-skew regime for the SIR redraw
uv run python experiments/posthoc/ablation.py --sizes small --families bernoulli --split test \
    --only raw isLaplace imhLaplace isSIR imhSIR --refresh-summaries
# acceptance-starved regime (does the absorbing-state fix move the huge-regime ECE?)
uv run python experiments/posthoc/ablation.py --sizes huge --families poisson bernoulli --split test \
    --only raw isLaplace imhLaplace imhAGQ imhSIR --refresh-summaries
# optional: re-check weight stability on other families/sizes
uv run python experiments/posthoc/newton_stability.py --families bernoulli --sizes large --n-datasets 128
```

Key comparisons to read: post-fix imhLaplace vs the committed full-run mds
(pre-fix `poisson_large.md` etc. — FFX/σ_rfx ECE and Corr R at huge); isAGQ vs
isLaplace σ_rfx ECE (expect the −0.046 → −0.027-ward shift); isSIR/imhSIR RFX
joint ECE + LOO-NLL on Bernoulli-small.

## Cluster results (2026-09-12, 512 datasets, s=1000, test split, GPU node)

Paired against the committed pre-fix full-run mds (same MB sample caches).

### The Newton robustness fix is the win

| Poisson-large | σ_rfx ECE pre-fix | post-fix | raw |
|---|---|---|---|
| isLaplace  | −0.046 | **−0.031** | −0.027 |
| imhLaplace | −0.070 | **−0.051** | −0.027 |

The isLaplace excess vs raw collapsed −0.019 → −0.004: **most of what the original
analysis attributed to "Laplace σ_rfx bias" was Newton-instability contamination of
the weights.** Bernoulli-huge weight health: max PSIS k 15.39 → 2.62 (fallback
40 % → 38 %). LOO-NLL unchanged at NUTS level (imh* 1.394–1.397 vs coldNuts 1.397).
Cost at s=1000: imhLaplace 0.4 → 0.7 s/ds.

### AGQ: no measurable benefit post-fix

isAGQ ≡ isLaplace to ~3 decimals on every metric at 512 datasets (σ_rfx ECE −0.031
both); imhAGQ ≈ imhLaplace (small σ_rfx EACE gain at huge: 0.032 vs 0.043). With the
weight contamination gone there is almost no residual integrated-likelihood bias for
AGQ to remove. **Keep `nagq` off by default** (option retained; 0.9–1.0 s/ds).

### SIR: small, consistent, in-direction — borderline at 1.7× cost

Largest at Bernoulli-huge: σ_rfx EACE 0.034 vs 0.043, RFX R 0.639 vs 0.632, joint
ECE −0.029 vs −0.031, LOO 0.409 vs 0.410. Bernoulli-small: σ_rfx EACE 0.023 vs
0.030, RFX ECE −0.021 vs −0.025. Never worse. **Keep `redraw='sir'` off by default**
(1.2 s/ds); revisit if per-group calibration at huge becomes a priority.

### Confirmed: the huge-regime FFX under-dispersion is the finite pool

imhLaplace FFX ECE at Bernoulli-huge: −0.047 pre-fix, −0.047 post-fix. Matches the
Normal-huge exact-target control — no weight/conditional upgrade touches it. Fixing
it needs proposal-side work (bigger pools, tempering, or SNIS with PSIS truncation).

## Status

- [x] Phases 0–3 + cluster scale-up: **done, defaults decided**
- Adopted: robustified `laplaceRfxModes` (backtracking + adaptive budget + 1-nat
  pinning guard) — now simply what `isLaplace`/`imhLaplace` do
- Retained as opt-in, off by default: `nagq` (AGQ weights), `redraw='sir'`
- Open (separate work item): huge-regime finite-pool under-dispersion; paper
  appendix numbers for MB+IS / MB-IMH on GLMMs predate the fix and improve slightly
  on re-run
