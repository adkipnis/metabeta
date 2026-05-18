Normal GLMM Plan
================

Last updated: 2026-05-18

Goal
----

Fast, prior-aware analytical summaries for Gaussian GLMMs. R-INLA is a slow reference,
not a backend. Keep the production path batched, simple, and in the low-millisecond range.

Retained Path
-------------

`glmm()` currently uses:

- raw Gaussian LMM initialization;
- marginal MAP refinement of β, diagonal σ_rfx, and σ_eps;
- reported β cap for `d > 4`: `β_report = clamp(β_MAP, ν_ffx ± 4τ_ffx)`;
- uncapped MAP β for BLUP residuals;
- diagonal final Ψ for GLS/BLUP recompute;
- one-shot posterior-moment EB update for diagonal σ_rfx.

The `d > 4` gate is intentional: MAP β carry-forward improves medium/large/huge rows but
overfits small low-d rows. The β cap is reporting-only; BLUP accuracy is protected by
continuing to use the uncapped MAP β internally.

Current Performance
-------------------

First 1000 datasets per row. Lower NRMSE is better.

| Dataset | part | EB FFX | EB σ | EB σ_eps | EB BLUP | EB ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1095 | 0.4203 | 0.2151 | 0.4173 | 2.52 |
| small-n-sampled | valid | 0.2588 | 0.5646 | 0.2169 | 0.5125 | 2.50 |
| small-n-sampled | test | 0.2827 | 0.4684 | 0.2169 | 0.4924 | 2.54 |
| medium-n-mixed | train | 0.2515 | 0.3619 | 0.1655 | 0.4198 | 3.05 |
| medium-n-sampled | valid | 0.2766 | 0.4131 | 0.1891 | 0.5151 | 3.71 |
| medium-n-sampled | test | 0.2623 | 0.3964 | 0.1949 | 0.4417 | 3.63 |
| large-n-mixed | train | 0.4075 | 0.3711 | 0.1268 | 0.4148 | 4.23 |
| large-n-sampled | valid | 0.3009 | 0.4316 | 0.1563 | 0.5069 | 4.39 |
| large-n-sampled | test | 0.3579 | 0.4415 | 0.1513 | 0.5126 | 4.66 |
| huge-n-mixed | train | 0.3314 | 0.3776 | 0.1161 | 0.4545 | 5.34 |
| huge-n-sampled | valid | 0.4485 | 0.3694 | 0.1375 | 0.4574 | 6.43 |
| huge-n-sampled | test | 0.3398 | 0.3870 | 0.1438 | 0.4619 | 6.63 |

R-INLA Reference
----------------

Mixed/train first-1000 rows with diagonal R-INLA:

| Dataset | EB FFX | σ-grid FFX | INLA FFX | EB σ | INLA σ | EB BLUP | INLA BLUP | EB ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1095 | 0.1095 | 0.0985 | 0.4203 | 0.3665 | 0.4173 | 0.4081 | 2.52 | 2.406 |
| medium-n-mixed | 0.2515 | 0.2283 | 0.2301 | 0.3619 | 0.3419 | 0.4198 | 0.4289 | 3.05 | 2.604 |
| large-n-mixed | 0.4075 | 0.2630 | 0.2377 | 0.3711 | 0.3393 | 0.4148 | 0.4185 | 4.23 | 2.786 |
| huge-n-mixed | 0.3314 | 0.2799 | 0.2413 | 0.3776 | 0.2808 | 0.4545 | 0.4548 | 5.34 | 2.965 |

Interpretation:

- INLA still has the strongest σ_rfx estimates, especially huge mixed.
- BLUP is already tied or slightly better analytically on medium/large/huge rows.
- The remaining actionable gap is reported FFX in rare ill-conditioned high-d rows.
- R-INLA is roughly hundreds of times slower, so it remains a reference only.

β Sigma-Grid Candidate
----------------------

The only retained experimental β-reporting patch is `normal_beta_sigma_grid=True`.
It is reporting-only and cap-hit gated:

`β_report = Σ_s w_s β(σ_rfx * scale_s)`, with scales `{0.75, 1.0, 1.3333333}` and
weights from the marginal target.

First-1000 required normal rows:

| Dataset | part | EB FFX | σ-grid FFX | σ-grid ms |
| --- | --- | ---: | ---: | ---: |
| small-n-mixed | train | 0.1095 | 0.1095 | 3.50 |
| small-n-sampled | valid | 0.2588 | 0.2588 | 2.66 |
| small-n-sampled | test | 0.2827 | 0.2827 | 2.62 |
| medium-n-mixed | train | 0.2515 | 0.2283 | 4.02 |
| medium-n-sampled | valid | 0.2766 | 0.2626 | 4.84 |
| medium-n-sampled | test | 0.2623 | 0.2594 | 4.86 |
| large-n-mixed | train | 0.4075 | 0.2630 | 5.55 |
| large-n-sampled | valid | 0.3009 | 0.2994 | 6.06 |
| large-n-sampled | test | 0.3579 | 0.2878 | 6.41 |
| huge-n-mixed | train | 0.3314 | 0.2799 | 7.03 |
| huge-n-sampled | valid | 0.4485 | 0.4448 | 8.34 |
| huge-n-sampled | test | 0.3398 | 0.3037 | 8.56 |

Decision: keep σ-grid opt-in until the full 8k comparison is run. Remove tail-grid and
direct cap-shrink heuristics from the implementation; they were diagnostics, not retained
paths.

Sigma-Grid Variant Sweep
------------------------

Tested 2026-05-18 on first-1000 required normal rows and on the 32 saved INLA tail rows
selected from the mixed/train diagnostic. Lower FFX is better; `ms/ds` is from the
first-1000 required benchmark.

Mixed/train first-1000 rows:

| Variant | small FFX | medium FFX | large FFX | huge FFX | small ms | medium ms | large ms | huge ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| map scalar | 0.1095 | 0.2283 | 0.2630 | 0.2799 | 3.34 | 3.96 | 5.52 | 6.96 |
| post-EB scalar | 0.1095 | 0.2284 | 0.2630 | 0.2799 | 3.37 | 3.82 | 5.41 | 6.57 |
| post-EB axis | 0.1095 | 0.2283 | 0.2630 | 0.2799 | 7.15 | 4.02 | 5.54 | 7.07 |
| post-EB ratio | 0.1095 | 0.2283 | 0.2630 | 0.2799 | 3.38 | 3.61 | 5.03 | 6.31 |

Saved INLA tail rows:

| Variant | rows | N | FFX | FFX-INLA | σ | σ-INLA | BLUP | BLUP-INLA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| map scalar | all | 32 | 0.3892 | +0.1214 | 0.1385 | +0.0292 | 0.2707 | +0.0130 |
| map scalar | cap-hit | 23 | 0.2494 | +0.1352 | 0.0996 | +0.0158 | 0.1689 | +0.0036 |
| post-EB scalar | all | 32 | 0.3892 | +0.1214 | 0.1385 | +0.0292 | 0.2707 | +0.0130 |
| post-EB axis | all | 32 | 0.3892 | +0.1213 | 0.1385 | +0.0292 | 0.2707 | +0.0130 |
| post-EB ratio | all | 32 | 0.3892 | +0.1214 | 0.1385 | +0.0292 | 0.2707 | +0.0130 |

Result: none of the post-EB, axis, or ratio variants closes the residual INLA gap. The
axis grid adds cost without accuracy benefit. The ratio grid is slightly worse on
`huge-n-sampled valid` (`0.4631` vs `0.4448` FFX). Do not promote axis or ratio. If we
keep a sigma-grid path, the scalar grid is still the best accuracy/complexity tradeoff;
post-EB scalar is tied but does not reduce the INLA gap. The failed variant branches were
removed from code; keep only the map-stage scalar sigma-grid candidate.

Next Steps
----------

1. Run the full 8k Normal benchmark with the scalar sigma-grid before promoting it.
2. Do not reintroduce axis, ratio, or post-EB grid branches unless a later diagnostic finds
   a new tail pattern where scalar averaging is not enough.
3. Curvature-aware β shrinkage was tested and removed. It shrank cap-hit, high-d rows
   toward `nu_ffx` based on conditional Gaussian posterior variance, but it was slower and
   did not improve over scalar sigma-grid:
   - curvature only: large mixed FFX `0.3699`, huge mixed `0.3214`;
   - sigma-grid + curvature, power `1.0`: large mixed `0.2637`, huge mixed `0.2807`;
   - sigma-grid + curvature, power `0.5`: large mixed `0.2633`, huge mixed `0.2804`;
   - scalar sigma-grid reference: large mixed `0.2630`, huge mixed `0.2799`.
4. Next candidate: improve σ_rfx EB directly. INLA's most stable remaining advantage is
   variance-scale accuracy, so test a small per-dimension log-σ refinement around the
   current EB moment estimate before attempting any richer β posterior correction.
5. If σ_rfx refinement does not help, revisit fixed-effect posterior mean correction, but
   not as direct shrink-to-prior; the failed curvature shrink suggests that the missing
   behavior is not a simple reliability scalar.
6. Avoid broad posterior machinery, multi-starts, EP, full PyTorch INLA, or NPE-context
   ablations for this analytical phase.

Commands
--------

Use `python -u` for all long analytical runs so completed blocks stream immediately.

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family n --methods current raw --max-datasets 1000 --batch-size 32

uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family n --methods current --max-datasets 1000 --batch-size 32 \
    --normal-beta-sigma-grid \
    --normal-beta-sigma-grid-scales 0.75 1.0 1.3333333

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-mixed,medium-n-mixed,large-n-mixed,huge-n-mixed \
    --partition train --n-epochs 2 --n-inla 1000 --n-total 1000 \
    --analytical-methods normal_eb,normal_sigma_grid --re-correlation diagonal

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-sampled,medium-n-sampled,large-n-sampled,huge-n-sampled \
    --partition valid --n-inla 1000 --n-total 1000 \
    --analytical-methods normal_eb,normal_sigma_grid --re-correlation diagonal

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical tests
```

Retired Lines
-------------

- R-INLA backend or full PyTorch INLA: incompatible with the throughput target.
- Standalone MAP option: EB is the retained Normal answer.
- Tail-grid and direct cap-shrinkage modes: not better enough to carry implementation
  branches.
- Final correlated Ψ for BLUP: estimated correlations are noisy and harmful.
