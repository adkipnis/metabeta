Normal GLMM Plan
================

Last updated: 2026-05-18

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

Full 8k required benchmark before the tail β correction was enabled:

| Dataset | part | FFX | σ | BLUP | ms | guard |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1859 | 0.3593 | 0.3745 | 3.44 | 0.000 |
| small-n-sampled | valid | 0.1486 | 0.4950 | 0.4614 | 3.21 | 0.000 |
| small-n-sampled | test | 0.2315 | 0.4772 | 0.4609 | 3.34 | 0.000 |
| medium-n-mixed | train | 0.2190 | 0.3062 | 0.4023 | 5.52 | 0.000 |
| medium-n-sampled | valid | 0.2510 | 0.4000 | 0.4823 | 5.67 | 0.000 |
| medium-n-sampled | test | 0.2530 | 0.4412 | 0.5030 | 5.76 | 0.000 |
| large-n-mixed | train | 0.2149 | 0.3162 | 0.4072 | 7.18 | 0.000 |
| large-n-sampled | valid | 0.2870 | 0.3781 | 0.5190 | 7.40 | 0.000 |
| large-n-sampled | test | 0.2721 | 0.3602 | 0.5022 | 7.70 | 0.000 |
| huge-n-mixed | train | 0.2404 | 0.3029 | 0.4263 | 9.99 | 0.003 |
| huge-n-sampled | valid | 0.3026 | 0.3631 | 0.4906 | 11.67 | 0.002 |
| huge-n-sampled | test | 0.2704 | 0.3630 | 0.4898 | 11.49 | 0.003 |

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

Tail Diagnostics
----------------

The 2026-05-18 mixed/train diagnostic scanned 8000 medium/large/huge rows and ran
diagonal R-INLA on the 16 worst analytical FFX rows per size.

| Tail set | N | previous β RMSE | tail β RMSE | INLA row RMSE |
| --- | ---: | ---: | ---: | ---: |
| medium-n-mixed | 16 | 0.7870 | 0.7870 | 0.5426 |
| large-n-mixed | 16 | 0.4441 | 0.3790 | 0.1956 |
| huge-n-mixed | 16 | 0.5304 | 0.4462 | 0.2310 |

The large/huge tail signal is real but narrow: INLA's β posterior-mean shift is aligned
with the analytical β error, while broad replacement worsened population accuracy. Keep
the damped scalar gate; do not reintroduce axis, ratio, curvature, or hard-shrink variants
without a new diagnostic showing a stable failure signature.

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

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical tests
```

Retired Lines
-------------

- R-INLA backend or full PyTorch INLA: incompatible with the throughput target.
- Standalone MAP option: EB is the retained Normal answer; MAP is only an internal stage.
- Axis, ratio, post-EB, curvature, hard-shrink, and broad tail-grid β variants.
- Final correlated Ψ for BLUP: estimated correlations are noisy and harmful here.
