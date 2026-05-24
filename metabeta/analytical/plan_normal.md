Normal GLMM Plan
================

Last updated: 2026-05-24 (Direction E adopted; D/F/G diagnosed and retired; H adopted; I/J diagnosed and retired)

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

Mixed/train first-1000 rows with diagonal R-INLA:

| Dataset | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | current ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.0913 | 0.0774 | 0.3401 | 0.3102 | 0.3850 | 0.3818 | 12.97 | 9.903 |
| medium-n-mixed | 0.2424 | 0.2352 | 0.3156 | 0.2975 | 0.4127 | 0.4157 | 11.75 | 8.379 |
| large-n-mixed | 0.2216 | 0.2015 | **0.2674** | 0.2713 | 0.3930 | 0.3962 | 15.50 | 8.858 |
| huge-n-mixed | 0.2381 | 0.2156 | 0.3014 | 0.2398 | 0.4057 | 0.4071 | 16.83 | 10.165 |

- INLA still has a σ_rfx edge on small/medium/huge; large-n-mixed now beats INLA after Direction A.
- BLUP is tied or slightly better analytically on medium/large/huge mixed rows.
- The FFX gap is concentrated in rare high-d, ill-conditioned tails (see `experiments/analytical/glmm_inla_results.md`).
- R-INLA is seconds per dataset; the analytical path is milliseconds.

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

**Direction I — Skew-corrected β marginal mean (DIAGNOSED and RETIRED 2026-05-24)**

Implemented 3rd-moment skewness correction in `_normalSigmaGridBetaAverage`: after computing
the softmax-weighted mean `mu = Σ_k w_k β_k`, add `0.25 * m3 / m2` where m3 and m2 are the
3rd and 2nd central moments of the grid β values under the posterior weights.

Result: null — zero effect on all sizes. Root cause: the scale grid
`{0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0}` is symmetric in log space, so for a
well-centered σ posterior (which EB ensures), the weights are approximately symmetric and
m3 ≈ 0 by construction. The correction is structurally zero whenever the EB σ is near the
posterior mode, which is exactly when the tail grid runs.

Infrastructure retained as `skew_correct=True` flag in `_normalSigmaGridBetaAverage` and
`normal_beta_tail_grid_skew_correct` kwarg, and `normal_direction_i` benchmark method.

**Direction J — σ-only REML-Newton polish after Adam (DIAGNOSED and RETIRED 2026-05-24)**

Implemented σ-only REML-Newton polish: after Adam's 20 steps, run 5 REML-Newton steps
for σ_rfx while holding β frozen at Adam's output (`freeze_beta=True`, `step_clamp=0.3`).
The goal was to tighten σ convergence without touching β (avoiding G's β-jump failure mode).

Result: universal regression — σ_rfx +7–55% worse across all sizes. Root cause: profile
REML-Newton at fixed β converges to the σ that maximises the marginal likelihood for
*that specific β*, not the joint MAP σ. The two objectives diverge because the marginal
REML accounts for β uncertainty via the `A^{-1}` precision term, whereas the profile REML
treats β as known. For small-n where β uncertainty is large, this mismatch is severe (+55%
σ, +41% FFX on small-n-sampled-test). For large-n the mismatch is smaller (+7–25% σ) but
still regresses consistently. The accept rate for the subsequent EB step jumps from ~3% to
~68% — the Newton-displaced σ triggers spurious EB "improvements".

Same structural failure mode as Direction G: β and σ objectives are coupled and cannot be
optimized independently. The retained infrastructure (`freeze_beta`, `step_clamp` in
`_normalBlockCoordMap`) is available for future experiments but defaults remain False/0.5.

**Direction H — Iterated MAP→EB loop (DONE 2026-05-24)**

2 outer iterations of `MAP-Adam(20) → moment-EB → grid-refine`; tail β only on last
iteration. BLUP β anchored to iteration 1's MAP value so warm-started iter-2 β doesn't
degrade BLUPs. Results on 500 datasets:

- FFX: −1% to −7% improvement across all sizes (largest gains on large/huge).
- BLUP: neutral (±0.2%) everywhere; huge-n-mixed regression (+5.3%) eliminated by anchor fix.
- σ_rfx: neutral.
- Cost: `+4–8 ms/ds` (~1.7× total; well within 10× headroom).

Default: `normal_map_outer_iterations=2` (previously 1).

**Direction G — Newton polish after Adam (DIAGNOSED and RETIRED 2026-05-24)**

Implemented `_normalBlockCoordMap`: block-coordinate Newton with exact GLS-MAP β +
REML-Newton σ_rfx (log-space step, ±0.5 clamp) + EM σ_eps. Two variants tested:
(1) pure Newton from LMM init (5 steps): σ NRMSE 2.1–3.1 vs 0.39–0.47 for current —
severely diverges because the REML score is large far from the MAP.
(2) Newton polish after Adam (3 steps from Adam's output): FFX +45% / σ +85% regression on
small-n; smaller regressions on larger sizes.

Root cause: step 1 of block-coord Newton replaces Adam's β with exact GLS-MAP β. This
changes residuals substantially, causing the REML score to be nonzero and the Newton σ
step to overshoot — same structural mechanism as Direction D (ill-conditioned rows collapse
GLS β toward prior mean, changing residuals). The warm-started Newton from Adam does not
converge in 3 steps.

Infrastructure retained in `map.py` (`_normalBlockCoordMap`, `use_newton` flag in
`refineNormalMapSrfx`) and benchmark method `normal_direction_g`, but defaults remain
False.

**Direction E — Wider 7-pt σ grid (DONE 2026-05-24)**

Replaced the 3-pt scalar grid `{0.75, 1.0, 1.333}` with `{0.5, 0.667, 0.833, 1.0, 1.2, 1.5, 2.0}`
for all three grid scales (`normal_laplace_eb_sigma_grid_scales`, `normal_beta_sigma_grid_scales`,
`normal_beta_tail_grid_scales`). Results on first-1000 rows: σ NRMSE improved 1–8% across
all sizes (huge-n-mixed: -8.1%), FFX neutral. Cost: `+0.5–1 ms/ds`.

Also added a `σ_eps_map` fix: `refineNormalMapSrfx` now stores Adam's MAP σ_eps in
`stats['normal_map_sigma_eps']`, and the tail-grid GLS uses it instead of the initial
projection σ_eps. Slight additional FFX improvement for gated-row tail corrections.

**Direction D — Unconditional β averaging (DIAGNOSED and RETIRED 2026-05-24)**

Attempted two implementations: (1) unconditional `_normalSigmaGridBetaAverage` in
`refineNormalMapSrfx`, (2) unconditional `maybe_correct_beta_tail` in `refineNormalLaplaceEb`
(lowering `beta_tail_grid_min_d` to 4 with `blend=0.25` and `blend=1.0`).

All variants regressed FFX on medium rows (+3–50%) while being neutral on large/huge.
Root cause: the GLS β at EB σ is worse than the Adam MAP β for the dominant gate group
(high-condition rows, ~65% of large/huge). For these rows, `A_data` is ill-conditioned
and the GLS collapses toward the prior mean; Adam's trajectory-aware β is better.
The existing `blend=0.25` cap-gated tail correction was already the sweet spot.

Direction D infrastructure code is retained in `refineNormalMapSrfx`/`refineNormalLaplaceEb`
as opt-in flags (`beta_sigma_grid_unconditional`, `beta_tail_grid_unconditional`) but
defaults remain False.

**Direction F — Cartesian σ_rfx grid for q=2 β averaging (RETIRED 2026-05-24)**

Not applicable: Cartesian β averaging is hampered by the same high-condition-row
mechanism as Direction D. No benefit without a working unconditional β averaging path.
Infrastructure code (`cartesian` flag in `_normalSigmaGridBetaAverage`) retained but
defaults False.

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

INLA still wins on mixed rows for both FFX and σ_rfx:

| Row | FFX gap | σ_rfx gap | notes |
| --- | ---: | ---: | --- |
| small-n-mixed | +0.031 (+40%) | +0.076 (+25%) | low-N rows, β cap fires 0%; tail-grid fires 0% |
| medium-n-mixed | -0.007 (-3%) | +0.048 (+16%) | already at parity on FFX; σ improved with E |
| large-n-mixed | +0.056 (+28%) | +0.083 (+31%) | β tail-grid gate fires 69.2% of rows |
| huge-n-mixed | +0.051 (+24%) | +0.081 (+38%) | tail-grid gate fires 76.1% of rows |

Direction E reduced the σ_rfx gap by ~20–25% vs INLA across all sizes. FFX gap unchanged.
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
    --data-ids small-n-mixed,medium-n-mixed,large-n-mixed,huge-n-mixed \
    --partition train --n-epochs 2 --n-inla 1000 --n-total 1000 \
    --analytical-methods normal_eb,current --re-correlation diagonal

# Save per-dataset row estimates for σ_rfx diagnostic (Normal family):
uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids medium-n-mixed,large-n-mixed,huge-n-mixed \
    --partition train --n-epochs 2 --n-inla 1000 --n-total 1000 \
    --analytical-methods normal_eb,current --re-correlation diagonal \
    --family n --save-inla-rows-dir experiments/analytical/inla_runs/normal_rows

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
