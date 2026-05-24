Normal GLMM Plan
================

Last updated: 2026-05-24 (Direction A implemented; debloat pass)

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

Pending Architecture Directions
--------------------------------

**Direction A — Softmax-weighted σ_rfx posterior mean (DONE 2026-05-24)**

`_normalSigmaRfxGridRefine` replaced argmax with softmax-weighted mean — same pattern as
`_normalSigmaGridBetaAverage` for β. σ posteriors are right-skewed so mean > mode; this
closed most of the INLA gap on σ_rfx.

**Direction B — Analytical β-profiling (not warranted)**

Diagnostic confirmed MAP σ already matches INLA mode; the gap was purely mean-vs-mode.
Direction B not needed.

**Direction C — Trace-based df for final σ_eps (low priority)**

The EM loop computes `T = Σ_g tr[W_g Z_g'Z_g]`; applying it as a one-shot df correction
to the projection denominator (`df_eff = mx_rank + T`) would correct the collinearity bias
in σ_eps without EM noise. Modest value; not attempted.

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
