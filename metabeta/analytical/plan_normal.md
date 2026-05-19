Normal GLMM Plan
================

Last updated: 2026-05-19

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
  improvement;
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
| small-n-mixed | train | 0.1089 | 0.4002 | 0.2151 | 0.4156 | 3.42 | 0.000 |
| small-n-sampled | valid | 0.2608 | 0.5695 | 0.2169 | 0.5119 | 2.75 | 0.000 |
| small-n-sampled | test | 0.2828 | 0.4759 | 0.2169 | 0.4931 | 2.74 | 0.000 |
| medium-n-mixed | train | 0.2283 | 0.3617 | 0.1655 | 0.4197 | 4.17 | 0.000 |
| medium-n-sampled | valid | 0.2626 | 0.4186 | 0.1891 | 0.5145 | 5.09 | 0.000 |
| medium-n-sampled | test | 0.2594 | 0.3825 | 0.1949 | 0.4403 | 4.96 | 0.000 |
| large-n-mixed | train | 0.2582 | 0.3643 | 0.1268 | 0.4135 | 6.20 | 0.000 |
| large-n-sampled | valid | 0.2970 | 0.4159 | 0.1563 | 0.5045 | 6.52 | 0.000 |
| large-n-sampled | test | 0.2872 | 0.4346 | 0.1513 | 0.5126 | 6.95 | 0.000 |
| huge-n-mixed | train | 0.2677 | 0.3484 | 0.1161 | 0.4528 | 7.89 | 0.004 |
| huge-n-sampled | valid | 0.4240 | 0.3562 | 0.1375 | 0.4555 | 9.50 | 0.000 |
| huge-n-sampled | test | 0.2947 | 0.3689 | 0.1438 | 0.4604 | 9.90 | 0.002 |

Full 8k required benchmark with the current tail β default:

| Dataset | part | FFX | σ | σ_eps | BLUP | ms | β-tail | guard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2243 | 0.3740 | 0.1661 | 0.3974 | 3.68 | 0.000 | 0.000 |
| small-n-sampled | valid | 0.1507 | 0.4962 | 0.2043 | 0.4637 | 3.06 | 0.000 | 0.000 |
| small-n-sampled | test | 0.2313 | 0.4758 | 0.2031 | 0.4607 | 3.20 | 0.000 | 0.000 |
| medium-n-mixed | train | 0.2184 | 0.3275 | 0.1295 | 0.4195 | 5.47 | 0.000 | 0.000 |
| medium-n-sampled | valid | 0.2509 | 0.3993 | 0.1752 | 0.4826 | 5.71 | 0.000 | 0.000 |
| medium-n-sampled | test | 0.2524 | 0.4399 | 0.1735 | 0.5023 | 5.81 | 0.000 | 0.000 |
| large-n-mixed | train | 0.2114 | 0.3133 | 0.1117 | 0.4066 | 7.79 | 0.682 | 0.000 |
| large-n-sampled | valid | 0.2779 | 0.3781 | 0.1409 | 0.5178 | 7.90 | 0.648 | 0.000 |
| large-n-sampled | test | 0.2672 | 0.3590 | 0.1354 | 0.5009 | 8.14 | 0.642 | 0.000 |
| huge-n-mixed | train | 0.2299 | 0.3126 | 0.0935 | 0.4261 | 10.65 | 0.724 | 0.003 |
| huge-n-sampled | valid | 0.2920 | 0.3620 | 0.1187 | 0.4897 | 11.48 | 0.719 | 0.002 |
| huge-n-sampled | test | 0.2668 | 0.3627 | 0.1201 | 0.4890 | 11.25 | 0.716 | 0.003 |

R-INLA Reference
----------------

Mixed/train first-1000 rows with diagonal R-INLA:

| Dataset | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | current ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1089 | 0.0985 | 0.4002 | 0.3665 | 0.4156 | 0.4081 | 3.42 | 2.406 |
| medium-n-mixed | 0.2283 | 0.2301 | 0.3617 | 0.3419 | 0.4197 | 0.4289 | 4.17 | 2.604 |
| large-n-mixed | 0.2582 | 0.2377 | 0.3643 | 0.3393 | 0.4135 | 0.4185 | 6.20 | 2.786 |
| huge-n-mixed | 0.2677 | 0.2413 | 0.3484 | 0.2808 | 0.4528 | 0.4548 | 7.89 | 2.965 |

Interpretation:

- INLA still has the clearest σ_rfx edge, especially huge mixed.
- BLUP is already tied or slightly better analytically on medium/large/huge mixed rows.
- The remaining FFX gap is concentrated in rare high-d or ill-conditioned tails.
- R-INLA is seconds per dataset, while the analytical path is milliseconds.

Sampled first-1000 rows with diagonal R-INLA:

| Dataset | part | current FFX | INLA FFX | current σ | INLA σ | current BLUP | INLA BLUP | current ms | INLA s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-sampled | valid | 0.2608 | 0.2023 | 0.5695 | 0.4450 | 0.5119 | 0.4988 | 2.87 | 2.377 |
| small-n-sampled | test | 0.2828 | 0.2008 | 0.4759 | 0.4357 | 0.4931 | 0.4756 | 2.73 | 2.382 |
| medium-n-sampled | valid | 0.2626 | 0.2490 | 0.4186 | 0.4048 | 0.5145 | 0.5201 | 4.81 | 3.032 |
| medium-n-sampled | test | 0.2594 | 0.2594 | 0.3825 | 0.3419 | 0.4403 | 0.4506 | 5.35 | 3.020 |
| large-n-sampled | valid | 0.2970 | 0.2527 | 0.4159 | 0.3428 | 0.5045 | 0.5069 | 6.21 | 3.430 |
| large-n-sampled | test | 0.2872 | 0.2710 | 0.4346 | 0.3984 | 0.5126 | 0.5052 | 6.65 | 3.431 |
| huge-n-sampled | valid | 0.4240 | 0.3110 | 0.3562 | 4.4979 † | 0.4555 | 0.4598 | 10.27 | 3.784 |
| huge-n-sampled | test | 0.2947 | 0.2732 | 0.3689 | 0.3122 | 0.4604 | 0.4639 | 9.63 | 3.768 |

† `huge-n-sampled valid` has an INLA σ_rfx outlier in the second true-σ quartile. Do not
use that cell as evidence against the analytical σ path.

Tail Diagnostics
----------------

Tail scans (2026-05-18) on 8000 medium/large/huge rows confirm the remaining FFX gap is
concentrated in rare high-d, ill-conditioned rows with frequent β prior-cap hits and
frequent singular/near-singular fixed/random design, where INLA's posterior-mean shift is
aligned with the analytical β error. BLUP is tied in these rows. The retained damped scalar
gate handles the actionable portion; previous broader β corrections improved selected tails
while moving full-population metrics backward. Low-priority future work: a very narrow β-only
safeguard gated on high-d, prior-capped, singular/near-singular rows, blending at most 25%
toward the prior/scalar-grid posterior mean. Full tail diagnostic tables are in
`experiments/analytical/glmm_inla_results.md`.

Known Structural Limits
-----------------------

- σ_eps df denominator is biased upward when many predictors are near-collinear with Z
  (active df drops, remaining directions over-absorb cross-group variance).
- z_rank can be underestimated when a group has n_g ≤ q (jittered inverse applied but
  rank counter may still inflate the df denominator, biasing σ_eps downward).

No correction is implemented; both are structural constraints of the one-pass projection
estimator.

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

uv run python -u experiments/analytical/glmm_normal_inla_diagnostic.py \
    --data-ids medium-n-mixed large-n-mixed huge-n-mixed --partition train \
    --n-epochs 2 --tail-scan 8000 --tail-k 16 --tail-metric ffx_eb_rmse \
    --methods current --batch-size 32 \
    --output-csv experiments/analytical/normal_ffx_tail_posterior_shift.csv

uv run python -u experiments/analytical/glmm_normal_inla_diagnostic.py \
    --data-ids large-n-sampled huge-n-sampled --partition valid \
    --tail-scan 8000 --tail-k 16 --tail-metric ffx_eb_rmse \
    --methods current --batch-size 32 \
    --output-csv experiments/analytical/normal_sampled_valid_ffx_tail_diagnostic.csv

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical tests
```

Retired Lines
-------------

- R-INLA backend or full PyTorch INLA: incompatible with the throughput target.
- Standalone MAP option: EB is the retained Normal answer; MAP is only an internal stage.
- Output-local MAP for final BLUP: oracle tests confirmed diagonal Ψ from MAP σ_rfx beats
  output-local MAP and full Ψ recompute for BLUP accuracy.
- Axis, ratio, post-EB, curvature, hard-shrink, and broad tail-grid β variants.
- Final correlated Ψ for BLUP: estimated correlations are noisy and harmful here.
