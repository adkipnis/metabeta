Normal GLMM Plan
================

Last updated: 2026-05-24 (Direction A implemented; debloat pass; D–J architectural patches proposed)

Goal
----

Fast, prior-aware analytical summaries for Gaussian GLMMs. R-INLA is a slow
accuracy reference, not a backend. The retained path should stay batched,
millisecond-scale, and simple enough to trust.

Default Path
------------

`glmm(..., likelihood_family=0)` now runs Normal EB by default:

- raw Gaussian LMM initialization;
- marginal MAP refinement of β, diagonal σ_rfx, and σ_eps;
- reported β cap for `d > 4`: `clamp(β_MAP, ν_ffx ± 4τ_ffx)`;
- uncapped MAP β for BLUP residuals;
- diagonal final Ψ for GLS/BLUP recompute;
- scalar β sigma-grid reporting over σ_rfx scales `{0.75, 1.0, 1.3333333}`;
- one-shot posterior-moment EB update for diagonal σ_rfx;
- one-pass coordinate σ_rfx grid over the same scales, accepted only on marginal-target
  improvement, reporting softmax-weighted posterior mean (Direction A);
- damped tail β correction for `d >= 9`, gated by β cap/stabilization or weak β
  precision and blended `25%` toward the scalar-grid posterior mean;
- rare BLUP/sigma guard for high-d aliased rows with implausibly large BLUP norms.

The β cap and tail correction are reporting-only. BLUPs continue to use the uncapped MAP
β unless the rare high-alias guard fires.

Current Performance
-------------------

First 1000 datasets per row with the default path. Lower NRMSE is better.

| Dataset | part | FFX | σ | σ_eps | BLUP | ms | guard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1093 | 0.3973 | 0.2151 | 0.4161 | 8.50 | 0.000 |
| small-n-sampled | valid | 0.2605 | 0.5627 | 0.2169 | 0.5120 | 3.21 | 0.000 |
| small-n-sampled | test | 0.2827 | 0.4639 | 0.2169 | 0.4923 | 3.06 | 0.000 |
| medium-n-mixed | train | 0.2283 | 0.3522 | 0.1655 | 0.4192 | 4.62 | 0.000 |
| medium-n-sampled | valid | 0.2626 | 0.3986 | 0.1891 | 0.5142 | 5.66 | 0.000 |
| medium-n-sampled | test | 0.2594 | 0.3798 | 0.1949 | 0.4405 | 5.53 | 0.000 |
| large-n-mixed | train | 0.2582 | 0.3608 | 0.1268 | 0.4135 | 6.89 | 0.000 |
| large-n-sampled | valid | 0.2971 | 0.4118 | 0.1563 | 0.5043 | 7.37 | 0.000 |
| large-n-sampled | test | 0.2870 | 0.4258 | 0.1513 | 0.5119 | 7.74 | 0.000 |
| huge-n-mixed | train | 0.2677 | 0.3490 | 0.1161 | 0.4529 | 7.99 | 0.004 |
| huge-n-sampled | valid | 0.4239 | 0.3502 | 0.1375 | 0.4556 | 9.81 | 0.000 |
| huge-n-sampled | test | 0.2947 | 0.3704 | 0.1438 | 0.4602 | 9.88 | 0.002 |

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
| small-n-mixed | +0.032 (+41%) | +0.087 (+28%) | low-N rows, β cap fires 0%; tail-grid fires 0% |
| medium-n-mixed | -0.007 (-3%) | +0.055 (+18%) | already at parity on FFX |
| large-n-mixed | +0.057 (+28%) | +0.090 (+33%) | β tail-grid gate fires 68.7% of rows |
| huge-n-mixed | +0.052 (+24%) | +0.109 (+46%) | tail-grid gate fires 76.0% of rows |

Two structural observations:

1. **σ_rfx mean-vs-mode is closed by Direction A only when the grid hits the right scale.**
   The default 3-point scalar grid `{0.75, 1.0, 1.333}` (i.e. ±33%) is too narrow to capture
   the right tail of π(σ_rfx | y) on diffuse posteriors. INLA typically uses 9–13 quadrature
   points along each axis. The remaining σ_rfx mean-vs-mode gap is a *grid resolution* issue,
   not a method-of-estimation issue.

2. **β posterior averaging machinery already exists but is gated off for most rows.**
   `_normalSigmaGridBetaAverage` computes the INLA-style β posterior mean
   `β̂ = Σ_k w_k β̂_GLS(σ_k)` with softmax weights from the marginal target. It is
   currently only invoked (a) inside `refineNormalMapSrfx` when the β prior cap fires
   (rare on mixed rows), and (b) inside `refineNormalLaplaceEb` as a 25%-damped tail
   correction gated on `d ≥ 9` + cap-or-cond. β output for the median mixed row is still
   the MAP-Adam β, not the integrated β. This is the most likely source of the residual
   FFX gap on mixed rows.

Latency budget: current default is `4–10 ms/ds`; budget ceiling is `~100 ms/ds`. There is
roughly **10× headroom** to spend on accuracy.

Pending Architecture Directions
--------------------------------

Ranked by expected impact on the FFX gap vs cost. Items D and E are the priority
candidates — they re-use existing machinery and target the diagnosed gap directly.

**Direction D — Always-on σ-grid β averaging (priority 1)**

Replace the MAP β output with the softmax-weighted σ-grid β average for `d ≥ d_thresh`
(start with `d_thresh = 4`, the existing β-cap threshold). The mechanism is already
implemented in `_normalSigmaGridBetaAverage`; the change is to drop the cap/stabilization
gate and use it unconditionally as the primary β output.

- Current call site: `refineNormalMapSrfx` calls it only inside `if beta_sigma_grid` AND
  `stabilize = cap_components & (d_count >= 5)`. The mixed-row β-cap fire rate is `~0–6%`
  (table above), so >94% of rows never see β averaging.
- Proposed: lift the `cap_components` gate and average across the σ grid for every active
  dataset with `d_count >= 4`. Keep the cap as a hard clamp on the averaged output
  (defense for prior tails), but stop using cap-firing as the *gate* for averaging.
- Expected: closes most of the small/large/huge-n-mixed FFX gap (analogous to INLA's main
  trick: integrate over θ instead of plugging in θ̂).
- Cost: each grid point is one Woodbury GLS + one marginal target eval. With the current
  3-point grid this is `~2–3 ms/ds` extra. With Direction E (wider grid) it can be
  `~5–15 ms/ds`. Well inside budget.
- Risk: low. β averaging is mathematically the marginal posterior mean under the Laplace
  approximation; it cannot regress FFX in expectation when the grid covers the σ
  posterior reasonably. The 25% blend already exists; remove the blend and use the full
  average for d>=d_thresh.

**Direction E — Wider/finer σ-grid (priority 2; pairs with D)**

The default `{0.75, 1.0, 1.3333}` grid covers `±33%` of σ̂. For diffuse posteriors
(small-n, high-q, near-zero σ), the right-tail mass sits beyond `1.333·σ̂_MAP`.

- Proposed grid (log-spaced, 7 points): `{0.5, 0.66, 0.83, 1.0, 1.2, 1.5, 2.0}`. This is
  roughly ±100% and approximates the CCD axis-points INLA uses.
- Alternative (5 points): `{0.5, 0.71, 1.0, 1.41, 2.0}` — half the cost, still wider.
- Apply to both `normal_beta_sigma_grid_scales` (β averaging) and
  `normal_laplace_eb_sigma_grid_scales` (Direction A σ_rfx refinement).
- Cost: 5-pt → `~4–7 ms/ds`; 7-pt → `~6–12 ms/ds`. Both inside budget.
- Risk: prior literature retired "2.0× scale, G-gated" because at 8k it did not improve
  large/huge. Re-investigate now that Direction A averages instead of argmax-selects:
  argmax discards low-weight tail points, but the softmax mean uses them. A wider grid
  should help more once paired with D.

**Direction F — Per-component σ_rfx grid for β averaging (priority 3; q≥2 only)**

Scalar grid scales all σ_rfx components together. For q≥2, the β covariance depends on
each component's σ separately, and the right-tail of one component can matter
independently of the others.

- For `q=2`: 2-D Cartesian grid (e.g. 5×5 = 25 candidates per dataset). Cost is
  `q^S` so this only scales to `q≤2` on the wider grid; for `q≥3` keep scalar.
- Direction A (`_normalSigmaRfxGridRefine`) already iterates *one coordinate at a time*
  for σ_rfx; the proposal is to do a full *Cartesian* grid for β averaging, where
  each candidate carries its own posterior weight.
- Expected: helps q=2 rows specifically; q=1 rows are unchanged (no coordinate to grid
  over). q≥3 rows fall back to scalar (D).
- Cost: 25-candidate solve at q=2 is `~10–20 ms/ds` on huge rows. Inside budget.

**Direction G — Newton-Raphson MAP replacing Adam (priority 4; speed > accuracy)**

`refineNormalMapSrfx` uses Adam `n_steps=20, lr=0.03`. At the optimum, Adam never quite
converges (lr too large for tight tolerance) and the resulting MAP β has small but
non-zero error vs the true MAP. Replace with Newton-Raphson using the same Woodbury
analytical gradients already available in `_remlNewtonStep`.

- Expected: 3–5 Newton steps to tight convergence. Better-anchored σ-grid because the
  grid is centered at the actual MAP.
- Cost: each Newton step builds the same Woodbury inner system as Adam; total cost
  comparable or slightly less. Net speed-up of `~1–3 ms/ds`.
- Risk: requires Hessian-vector products or block-explicit Hessians. `torch.autograd`
  Hessian on `_logMarginalTarget` is expensive per-batch; a custom analytic form mirroring
  `_remlNewtonStep` (REML Newton for σ_rfx already implemented in `lmm.py`) is cheaper.

**Direction H — Iterated MAP→EB→MAP refinement loop (priority 5)**

Currently the refinement pipeline is one pass each: `MAP-Adam → moment-EB → grid-refine →
tail β`. Each stage outputs to the next; there is no fixed-point iteration. For
ill-conditioned high-d rows the first MAP solution biases the EB σ which then biases the
grid β.

- Proposed: 2–3 outer iterations of `(MAP β | σ) ↔ (EB σ | β)` until ‖Δ‖ < tol. Each
  outer iteration is one MAP and one EB.
- Cost: `+3–6 ms/ds` per extra outer.
- Risk: Adam's noisy convergence within iteration means small-step convergence might
  oscillate. Pairs naturally with Direction G (Newton MAP).

**Direction I — Skew-corrected β marginal mean (priority 6; speculative)**

The β marginal posterior π(β | y) integrated over σ can be skewed: the right tail of σ
inflates one β tail more than the other. INLA explicitly approximates this skew with a
Simpson-rule integration along each β coordinate. For Normal LMM the skew is small but
nonzero on diffuse-σ rows.

- Implementation: at each σ grid point, compute β posterior mean *and* a 3-point Simpson
  approximation of the β skewness contribution.
- Expected: small additional FFX improvement on small-n-mixed (highest-skew rows).
- Cost: `+1–2 ms/ds`. Speculative — implement only if D+E leave residual gap.

**Direction J — Block Newton joint update for (β, σ) (priority 7; speculative)**

For Normal LMM, the joint posterior is closed-form Gaussian conditional on σ. The Adam
joint update over (β, log σ_rfx, log σ_eps) couples optimization across blocks; a block
alternation with exact β-GLS step + Newton-σ step would converge faster per step. Same
flavor as Direction G but applied jointly.

- Implementation: alternate (1) exact `_normalGlsAndBlups` for β given σ; (2) Newton step
  on σ given β using REML score and Fisher info (already implemented for diagonal σ in
  `_remlNewtonStep`).
- Expected: tighter MAP convergence + cleaner downstream grids.
- Cost: comparable to current Adam; possibly faster if convergence is 3–5 iterations.
- Risk: large refactor of `refineNormalMapSrfx` interior.

Bernoulli/Poisson lessons that inform priorities
-------------------------------------------------

What worked elsewhere maps cleanly to D+E here:

- **Bernoulli EB closed FFX gap unconditionally** by switching from PQL-init β to a
  Laplace-EB refined β over diagonal σ. The equivalent for Normal is using the σ-grid
  β average (D) instead of the MAP β plug-in.
- **Poisson VG refinement on a wider σ grid** is the Poisson equivalent of D+E. The
  Poisson plan reports VG = `~11 ms/ds` per dataset on top of EB; Normal's analogue is
  much cheaper because the conditional posterior is closed-form Gaussian.
- **Multi-start, EP, full INLA: explicitly deferred** in both Bernoulli and Poisson plans
  as low-expected-value. The same applies here for any directions beyond H.
- **σ-only patches deprioritized in Poisson**: "true/INLA-σ plug-in diagnostics did not
  close the FFX gap". The asymmetry is the same here: σ_rfx accuracy alone does not close
  FFX unless β is also integrated over σ uncertainty (which is exactly D).

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
