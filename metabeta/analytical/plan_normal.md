Normal GLMM Plan
================

Last updated: 2026-05-24 (directions D–J investigated; E and H adopted; all others retired; tables merged and plan debloated)

Goal
----

Fast, prior-aware analytical summaries for Gaussian GLMMs. R-INLA is a slow
accuracy reference, not a backend. The retained path should stay batched,
millisecond-scale, and simple enough to trust.

Default Path
------------

`glmm(..., likelihood_family=0)` now runs Normal EB by default with 2 outer iterations:

**Outer loop × 2** (`normal_map_outer_iterations=2`; tail β only on last iteration):
- marginal MAP refinement of β, diagonal σ_rfx, and σ_eps (Adam, 20 steps);
- reported β cap for `d > 4`: `clamp(β_MAP, ν_ffx ± 4τ_ffx)`;
- scalar β sigma-grid reporting over σ_rfx scales `{0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0}`;
- one-shot posterior-moment EB update for diagonal σ_rfx;
- one-pass coordinate σ_rfx grid over the same 7-pt scales, accepted only on marginal-target
  improvement, reporting softmax-weighted posterior mean (Direction A + E);

**Final pass only:**
- damped tail β correction for `d >= 9`, gated by β cap/stabilization or weak β
  precision; blended `25%` toward the grid posterior mean computed with MAP σ_eps;
- rare BLUP/sigma guard for high-d aliased rows with implausibly large BLUP norms.

BLUP β is anchored to iteration 1's uncapped MAP β across both outer iterations, preventing
the warm-started second-pass β from drifting further from OLS. The β cap and tail correction
are reporting-only.

Current Performance
-------------------

First 1000 datasets per row with the default path. Lower NRMSE is better.

| Dataset | part | FFX | σ | σ_eps | BLUP | ms | guard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1091 | 0.3861 | 0.2151 | 0.4151 | 3.30 | 0.000 |
| small-n-sampled | valid | 0.2605 | 0.5618 | 0.2169 | 0.5117 | 2.62 | 0.000 |
| small-n-sampled | test | 0.2828 | 0.4647 | 0.2169 | 0.4923 | 2.61 | 0.000 |
| medium-n-mixed | train | 0.2283 | 0.3475 | 0.1655 | 0.4189 | 3.93 | 0.000 |
| medium-n-sampled | valid | 0.2625 | 0.3941 | 0.1891 | 0.5137 | 4.74 | 0.000 |
| medium-n-sampled | test | 0.2594 | 0.3673 | 0.1949 | 0.4399 | 4.58 | 0.000 |
| large-n-mixed | train | 0.2579 | 0.3542 | 0.1268 | 0.4126 | 5.69 | 0.000 |
| large-n-sampled | valid | 0.2972 | 0.3894 | 0.1563 | 0.5011 | 6.01 | 0.000 |
| large-n-sampled | test | 0.2863 | 0.4188 | 0.1513 | 0.5118 | 6.29 | 0.000 |
| huge-n-mixed | train | 0.2673 | 0.3209 | 0.1161 | 0.4519 | 7.26 | 0.004 |
| huge-n-sampled | valid | 0.4241 | 0.3382 | 0.1375 | 0.4542 | 8.72 | 0.000 |
| huge-n-sampled | test | 0.2940 | 0.3585 | 0.1438 | 0.4592 | 9.12 | 0.002 |

R-INLA Reference
----------------

N=1000 datasets with diagonal R-INLA. Mixed rows are train; sampled rows are valid/test (OOD). Times for mixed are batched; sampled are sequential/non-batched.

| Dataset | part | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | current ms | INLA s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.0913 | 0.0774 | 0.3401 | 0.3102 | 0.3850 | 0.3818 | 12.97 | 9.903 |
| medium-n-mixed | train | 0.2424 | 0.2352 | 0.3156 | 0.2975 | 0.4127 | 0.4157 | 11.75 | 8.379 |
| large-n-mixed | train | 0.2216 | 0.2015 | **0.2674** | 0.2713 | 0.3930 | 0.3962 | 15.50 | 8.858 |
| huge-n-mixed | train | 0.2381 | 0.2156 | 0.3014 | 0.2398 | 0.4057 | 0.4071 | 16.83 | 10.165 |
| small-n-sampled | valid | 0.2746 | 0.2151 | 0.5398 | 0.5313 | 0.4814 | 0.4755 | 59.54 | 9.853 |
| small-n-sampled | test | 0.2393 | 0.1675 | 0.4144 | 0.4119 | 0.4230 | 0.4156 | 58.47 | 9.910 |
| medium-n-sampled | valid | 0.2371 | 0.2296 | 0.4431 | 0.3201 | 0.5712 | 0.5733 | 102.87 | 8.329 |
| medium-n-sampled | test | 0.2209 | **0.2339** | 0.3347 | 0.3103 | 0.4373 | 0.4426 | 102.02 | 8.326 |
| large-n-sampled | valid | 0.2718 | 0.2389 | 0.3344 | 0.3226 | 0.4737 | 0.4769 | 118.74 | 8.897 |
| large-n-sampled | test | 0.2620 | 0.2514 | 0.3715 | 0.3601 | 0.4775 | 0.4726 | 125.06 | 8.877 |
| huge-n-sampled | valid | 0.3981 | 0.2907 | **0.3511** | 0.5085 | 0.4871 | 0.4897 | 138.55 | 10.143 |
| huge-n-sampled | test | 0.2554 | 0.2491 | 0.3259 | 0.2895 | 0.5531 | 0.5554 | 132.61 | 10.135 |

Bolded cells: current beats INLA (large-train σ_rfx; medium-test FFX by −6%; huge-valid σ_rfx by −31%). BLUP is tied everywhere (< 1% gap). R-INLA is 8–10 s/dataset; analytical path is milliseconds (~70–100× faster).

Known Structural Limits
-----------------------

- σ_eps df denominator is biased upward when many predictors are near-collinear with Z
  (active df drops, remaining directions over-absorb cross-group variance).
- z_rank can be underestimated when a group has n_g ≤ q (jittered inverse applied but
  rank counter may still inflate the df denominator, biasing σ_eps downward).

No correction is implemented; both are structural constraints of the one-pass projection
estimator.

Closed Directions (historical)
------------------------------

**Direction I — Skew-corrected β marginal mean (RETIRED 2026-05-24)**

Added 3rd-moment skewness correction to the softmax-weighted β average. Zero effect: the 7-pt log-spaced grid is symmetric so m3 ≈ 0 whenever EB σ is near the posterior mode — which is exactly when the correction would run.

**Direction J — σ-only REML-Newton polish after Adam (RETIRED 2026-05-24)**

After Adam, ran 5 REML-Newton steps for σ_rfx with β frozen. Universal regression (+7–55% σ across all sizes): profile REML at fixed β optimises the wrong objective — it treats β as known rather than uncertain, so the Newton-displaced σ diverges from the joint MAP σ and triggers spurious EB "improvements" (EB accept rate 3% → 68%). Same coupled-objectives failure as Direction G.

**Direction H — Iterated MAP→EB loop (ADOPTED 2026-05-24)**

2 outer iterations of `MAP-Adam(20) → moment-EB → grid-refine`; tail β only on last iteration. BLUP β anchored to iteration 1's MAP so warm-started iter-2 β doesn't degrade BLUPs. FFX −1–7% across all sizes; σ_rfx and BLUP neutral. Cost +4–8 ms/ds (~1.7× total). Default: `normal_map_outer_iterations=2`.

**Direction G — Newton polish after Adam (RETIRED 2026-05-24)**

Block-coordinate Newton (exact GLS-MAP β + REML-Newton σ_rfx + EM σ_eps) after Adam. Pure Newton from LMM init severely diverges; polish-after-Adam regresses FFX +45% / σ +85% on small-n. Step 1 replaces Adam's β with GLS β, which changes residuals substantially and causes the Newton σ step to overshoot via the same collapse-toward-prior mechanism as Direction D.

**Direction E — Wider 7-pt σ grid (DONE 2026-05-24)**

Replaced the 3-pt scalar grid `{0.75, 1.0, 1.333}` with `{0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0}`
for all three grid scales (`normal_laplace_eb_sigma_grid_scales`, `normal_beta_sigma_grid_scales`,
`normal_beta_tail_grid_scales`). Results on first-1000 rows: σ NRMSE improved 1–8% across
all sizes (huge-n-mixed: -8.1%), FFX neutral. Cost: `+0.5–1 ms/ds`.

Also added a `σ_eps_map` fix: `refineNormalMapSrfx` now stores Adam's MAP σ_eps in
`stats['normal_map_sigma_eps']`, and the tail-grid GLS uses it instead of the initial
projection σ_eps. Slight additional FFX improvement for gated-row tail corrections.

**Direction D — Unconditional β averaging (RETIRED 2026-05-24)**

Unconditional σ-grid β averaging at both MAP and EB stages. All variants regressed FFX on medium (+3–50%) while neutral on large/huge. Root cause: ~65% of large/huge rows are ill-conditioned — GLS β collapses toward the prior mean there, and Adam's β is better. The existing 25%-blended cap-gated tail correction is already the sweet spot.

**Direction F — Cartesian σ_rfx grid for q=2 β averaging (RETIRED 2026-05-24)**

Cartesian σ-grid β averaging for q=2. Not applicable: hampered by the same ill-conditioned-row mechanism as Direction D. No benefit without a working unconditional β averaging path.

**Direction A — Softmax-weighted σ_rfx posterior mean (DONE 2026-05-24)**

`_normalSigmaRfxGridRefine` replaced argmax with softmax-weighted mean — same pattern as
`_normalSigmaGridBetaAverage` for β. σ posteriors are right-skewed so mean > mode; this
closed most of the INLA gap on σ_rfx.

**Direction B — Analytical β-profiling (not warranted)**

Diagnostic confirmed MAP σ already matches INLA mode; the gap was purely mean-vs-mode.
Direction B not needed.

**Direction C — Trace-based df for final σ_eps (tried 2026-05-24; reverted)**

Attempted: replace `z_rank` with `T = Σ_g tr[W_g Z_g'Z_g]` in the projection df
denominator. Since T ≤ z_rank, corrected denominator is larger → σ_eps_corrected smaller.
Result: σ_eps NRMSE increased +19–117% across all configurations (FFX/σ_rfx/BLUP flat).
The correction reduces σ_eps for all datasets even when there is no collinearity, which
outweighs any benefit in the rare high-collinearity case. No further attempts planned.

Gap Diagnosis vs INLA (2026-05-24 review)
-----------------------------------------

Gap = current − INLA; positive = current worse. Mixed rows are train; sampled are valid/test.

| Row | part | FFX gap | σ_rfx gap | notes |
| --- | --- | ---: | ---: | --- |
| small-n-mixed | train | +0.031 (+40%) | +0.076 (+25%) | low-N; β cap and tail-grid fire 0% |
| medium-n-mixed | train | −0.007 (−3%) | +0.048 (+16%) | FFX parity; σ improved with E |
| large-n-mixed | train | +0.056 (+28%) | +0.083 (+31%) | tail-grid gate fires 69% |
| huge-n-mixed | train | +0.051 (+24%) | +0.081 (+38%) | tail-grid gate fires 76% |
| small-n-sampled | valid | +0.060 (+28%) | +0.009 (+2%) | σ_rfx nearly tied |
| small-n-sampled | test | +0.072 (+43%) | +0.003 (+1%) | σ_rfx tied |
| medium-n-sampled | valid | +0.008 (+3%) | +0.123 (+38%) | FFX near-parity; σ_rfx gap reappears |
| medium-n-sampled | test | **−0.013 (−6%)** | +0.024 (+8%) | current beats INLA on FFX |
| large-n-sampled | valid | +0.033 (+14%) | +0.012 (+4%) | small gaps on both |
| large-n-sampled | test | +0.011 (+4%) | +0.011 (+3%) | near-parity on both |
| huge-n-sampled | valid | +0.107 (+37%) | **−0.157 (−31%)** | current beats INLA on σ_rfx |
| huge-n-sampled | test | +0.006 (+2%) | +0.036 (+13%) | near-parity on FFX |

Direction E reduced σ_rfx gap by ~20–25% across all sizes. FFX gap unchanged.
The FFX gap is driven by high-condition rows (65%+ of large/huge train data) where the
GLS β collapses toward the prior — existing 25%-blended tail correction is already optimal.

Two structural observations:

1. **σ_rfx mean-vs-mode is closed by Direction A only when the grid hits the right scale.**
   The default 3-point scalar grid `{0.75, 1.0, 1.333}` (i.e. ±33%) is too narrow to capture
   the right tail of π(σ_rfx | y) on diffuse posteriors. INLA typically uses 9–13 quadrature
   points along each axis. The remaining σ_rfx mean-vs-mode gap is a *grid resolution* issue,
   not a method-of-estimation issue.

2. **β posterior averaging (Direction D) cannot unconditionally replace MAP-Adam β.**
   `_normalSigmaGridBetaAverage` computes the INLA-style β posterior mean
   `β̂ = Σ_k w_k β̂_GLS(σ_k)`. Unconditional use regressed FFX on medium rows (+3–50%)
   because ~65% of large/huge rows are ill-conditioned: `A_data` condition ≥ 1000, causing
   GLS to collapse toward the prior mean — Adam's trajectory-aware β is structurally better
   there. The existing 25%-blended cap-gated tail correction is already the sweet spot.
   The remaining FFX gap requires a *self-consistent (β, σ)* MAP estimate (Direction H/J),
   not just better σ averaging.

Latency budget: current default is `3–10 ms/ds`; budget ceiling is `~100 ms/ds`. There is
roughly **10× headroom** to spend on accuracy.

Pending Architecture Directions
--------------------------------

Directions D, E, F, G, H, I, J investigated and resolved (2026-05-24). No open directions remain.

Bernoulli/Poisson lessons that inform priorities
-------------------------------------------------

- **Bernoulli EB closed FFX gap unconditionally** by switching from PQL-init β to a
  Laplace-EB refined β over diagonal σ. For Normal, the analogous move (Direction D:
  σ-grid β average) was diagnosed and retired — it regresses on ill-conditioned rows where
  GLS collapses to the prior. The self-consistent MAP approach (Direction H: iterated
  MAP→EB — adopted) is the Normal-appropriate analogue. Direction J (block Newton) was
  diagnosed and retired: β and σ cannot be optimized with separate objectives.
- **Poisson VG refinement on a wider σ grid** is the Poisson equivalent of E. The
  Poisson plan reports VG = `~11 ms/ds` per dataset on top of EB; Normal's analogue
  (Direction E) is much cheaper because the conditional posterior is closed-form Gaussian.
  Direction E was adopted.
- **Multi-start, EP, full INLA: explicitly deferred** in both Bernoulli and Poisson plans
  as low-expected-value. The same applies here for any directions beyond H.
- **σ-only patches deprioritized in Poisson**: "true/INLA-σ plug-in diagnostics did not
  close the FFX gap". The asymmetry is the same here: σ_rfx accuracy alone does not close
  FFX. The FFX gap requires a tighter, self-consistent (β, σ) MAP estimate.

Commands
--------

Use `python -u` for long analytical runs so completed blocks stream immediately.

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family n --methods current raw --max-datasets 1000 --batch-size 32

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-mixed --partition train --n-total 1000 \
    --analytical-methods current --no-save-inla-rows

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical tests
```

Retired Lines
-------------

- R-INLA backend or full PyTorch INLA: incompatible with the throughput target.
- Standalone MAP option: EB is the retained Normal answer; MAP is only an internal stage.
- `mode='gradient'` in `refineNormalLaplaceEb`: gradient Adam loop for σ_rfx — removed
  in 2026-05-24 debloat pass; moment EB is the only retained mode.
- Output-local MAP for final BLUP: oracle tests confirmed diagonal Ψ from MAP σ_rfx beats
  output-local MAP and full Ψ recompute for BLUP accuracy.
- Axis, ratio, post-EB, curvature, hard-shrink, and broad tail-grid β variants.
- Final correlated Ψ for BLUP: estimated correlations are noisy and harmful here.
- Wider σ_rfx grid (2.0× scale, G-gated): 8k showed no improvement on large/huge; the
  standard EB + 1.333 grid already captures available marginal-target improvement.
- Per-dimension moment EB (moment_per_dim): too liberal — accepts individual-dim updates
  where the joint posterior is not improving; regressed small-n-sampled σ by +0.013.
- τ_rfx floor for W_g (moment_sigma_tau_floor): no material improvement; moment EB is
  already robust to poor initialization via the prior regularizer.
