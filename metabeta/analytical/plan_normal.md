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

Decision: scalar β sigma-grid remains the best fixed-effect reporting patch. The full 8k
comparison below confirms it is the right companion for direct σ_rfx grid. Remove
tail-grid and direct cap-shrink heuristics from the implementation; they were diagnostics,
not retained paths.

Direct σ_rfx Grid Candidate
---------------------------

Implemented as `normal_laplace_eb_sigma_grid_refine=True`. After the current Normal EB
moment update accepts or rejects its candidate, this runs a one-pass coordinate grid over
diagonal `σ_rfx` scales `{0.75, 1.0, 1.3333333}`. Each coordinate update is accepted only
when the exact Gaussian marginal target plus priors improves. β and σ_eps are fixed; final
GLS/BLUP is recomputed through the existing diagonal path.

First-1000 required normal rows:

| Dataset | part | EB FFX | σ-refine FFX | EB σ | σ-refine σ | EB BLUP | σ-refine BLUP | σ-refine ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1095 | 0.1089 | 0.4203 | 0.4002 | 0.4173 | 0.4156 | 3.80 |
| small-n-sampled | valid | 0.2588 | 0.2608 | 0.5646 | 0.5695 | 0.5125 | 0.5119 | 2.97 |
| small-n-sampled | test | 0.2827 | 0.2828 | 0.4684 | 0.4759 | 0.4924 | 0.4931 | 2.85 |
| medium-n-mixed | train | 0.2515 | 0.2515 | 0.3619 | 0.3617 | 0.4198 | 0.4197 | 4.12 |
| medium-n-sampled | valid | 0.2766 | 0.2766 | 0.4131 | 0.4186 | 0.5151 | 0.5145 | 4.90 |
| medium-n-sampled | test | 0.2623 | 0.2623 | 0.3964 | 0.3825 | 0.4417 | 0.4403 | 4.79 |
| large-n-mixed | train | 0.4075 | 0.4075 | 0.3711 | 0.3643 | 0.4148 | 0.4135 | 5.53 |
| large-n-sampled | valid | 0.3009 | 0.3009 | 0.4316 | 0.4159 | 0.5069 | 0.5045 | 6.02 |
| large-n-sampled | test | 0.3579 | 0.3579 | 0.4415 | 0.4346 | 0.5126 | 0.5126 | 6.32 |
| huge-n-mixed | train | 0.3314 | 0.3314 | 0.3776 | 0.3481 | 0.4545 | 0.4526 | 7.13 |
| huge-n-sampled | valid | 0.4485 | 0.4485 | 0.3694 | 0.3562 | 0.4574 | 0.4555 | 8.76 |
| huge-n-sampled | test | 0.3398 | 0.3398 | 0.3870 | 0.3680 | 0.4619 | 0.4604 | 9.19 |

Combined with `normal_beta_sigma_grid=True`, FFX remains equal to the β-grid path while
the σ/BLUP gains above are retained.

Full 8k required benchmark, β sigma-grid vs current β sigma-grid plus direct σ_rfx grid
and BLUP guard:

| Dataset | part | β-grid FFX | current FFX | β-grid σ | current σ | β-grid BLUP | current BLUP | β-grid ms | current ms | guard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1860 | 0.1859 | 0.3688 | 0.3593 | 0.3756 | 0.3745 | 3.60 | 3.44 | 0.000 |
| small-n-sampled | valid | 0.1485 | 0.1486 | 0.5212 | 0.4950 | 0.4617 | 0.4614 | 3.34 | 3.21 | 0.000 |
| small-n-sampled | test | 0.2316 | 0.2315 | 0.4724 | 0.4772 | 0.4608 | 0.4609 | 3.42 | 3.34 | 0.000 |
| medium-n-mixed | train | 0.2190 | 0.2190 | 0.3106 | 0.3062 | 0.4025 | 0.4023 | 5.85 | 5.52 | 0.000 |
| medium-n-sampled | valid | 0.2510 | 0.2510 | 0.4081 | 0.4000 | 0.4828 | 0.4823 | 5.85 | 5.67 | 0.000 |
| medium-n-sampled | test | 0.2530 | 0.2530 | 0.5073 | 0.4412 | 0.5049 | 0.5030 | 5.91 | 5.76 | 0.000 |
| large-n-mixed | train | 0.2149 | 0.2149 | 0.3400 | 0.3162 | 0.4091 | 0.4072 | 7.46 | 7.18 | 0.000 |
| large-n-sampled | valid | 0.2870 | 0.2870 | 0.3980 | 0.3781 | 0.5203 | 0.5190 | 7.50 | 7.40 | 0.000 |
| large-n-sampled | test | 0.2721 | 0.2721 | 0.3772 | 0.3602 | 0.5030 | 0.5022 | 7.83 | 7.70 | 0.000 |
| huge-n-mixed | train | 0.2404 | 0.2404 | 0.3205 | 0.3029 | 0.4272 | 0.4263 | 10.21 | 9.99 | 0.003 |
| huge-n-sampled | valid | 0.3026 | 0.3026 | 0.4041 | 0.3631 | 0.6752 | 0.4906 | 11.05 | 11.67 | 0.002 |
| huge-n-sampled | test | 0.2704 | 0.2704 | 0.3865 | 0.3630 | 0.4917 | 0.4898 | 10.89 | 11.49 | 0.003 |

Decision: keep the combined path as the strongest current Normal analytical candidate.
The direct σ_rfx grid improves σ NRMSE on 11/12 rows, and the BLUP guard fixes the
`huge-n-sampled valid` tail without affecting smaller rows. The guard fires only on huge
rows (`0.2-0.3%`), so the added logic is narrow.

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

Huge Sampled BLUP Tail
----------------------

Diagnostic run on all 8192 `huge-n-sampled valid` rows with the combined β-grid plus
direct σ_rfx grid path, selecting the 16 largest analytical BLUP RMSE rows and running
diagonal R-INLA on those rows:

- overall analytical BLUP NRMSE is dominated by a tiny tail: dropping dataset `1550`
  alone reduces NRMSE from `0.669` to `0.495`; dropping the top 16 reduces it to `0.447`;
- the top failure is not caused by the direct σ_rfx grid. β-grid only already has BLUP
  RMSE `6.52`; direct σ_rfx grid changes it to `6.52`;
- `σ_eps` is accurate in the top rows, but one or two `σ_rfx` dimensions inflate sharply;
  dataset `1550` has true σ `[0.119, 0.606, 0.017, 0.400]`, analytical
  `[0.153, 2.614, 0.120, 0.445]`, and INLA `[0.136, 0.624, 0.040, 0.397]`;
- BLUP norm inflation is the direct symptom: dataset `1550` true/analytical/INLA BLUP
  RMS is `0.353 / 6.530 / 0.349`; dataset `7580` is `0.344 / 1.693 / 0.292`;
- the common signature is high-d sampled design aliasing: `d >= 13`, max fixed/random
  design R2 `1.0`, and singular residual fixed-effect design after projecting out `Z`.

Implemented guard: for rare high-d rows where the final BLUP RMS and at least one σ_rfx
dimension exceed the random-effect prior scale by a large margin, cap only inflated
σ_rfx dimensions at `4 * tau_rfx` and recompute BLUPs using OLS residuals. Reported β is
unchanged. On the two known pathologies:

- dataset `1550`: BLUP RMSE `6.525 -> 0.088`;
- dataset `7580`: BLUP RMSE `1.744 -> 0.528`.

Next Steps
----------

1. Treat scalar β sigma-grid plus direct σ_rfx grid plus the BLUP guard as the current
   best Normal candidate. Keep flags explicit until the next R-INLA comparison is
   refreshed, then consider making this the default Normal EB path.
2. Refresh the Normal R-INLA comparison with the guarded current path, especially sampled
   valid/test rows, to check whether the `huge-n-sampled valid` INLA gap is now mostly
   closed.
3. Do not reintroduce axis, ratio, or post-EB grid branches unless a later diagnostic finds
   a new tail pattern where scalar averaging is not enough.
4. Curvature-aware β shrinkage was tested and removed. It shrank cap-hit, high-d rows
   toward `nu_ffx` based on conditional Gaussian posterior variance, but it was slower and
   did not improve over scalar sigma-grid:
   - curvature only: large mixed FFX `0.3699`, huge mixed `0.3214`;
   - sigma-grid + curvature, power `1.0`: large mixed `0.2637`, huge mixed `0.2807`;
   - sigma-grid + curvature, power `0.5`: large mixed `0.2633`, huge mixed `0.2804`;
   - scalar sigma-grid reference: large mixed `0.2630`, huge mixed `0.2799`.
5. Direct σ_rfx grid passed the 8k benchmark as a variance-scale companion to the β
   reporting grid, but it does not solve the FFX tail gap by itself.
6. If more FFX improvement is needed after the next INLA comparison, revisit fixed-effect
   posterior mean correction, but not as direct shrink-to-prior; the failed curvature
   shrink suggests that the missing behavior is not a simple reliability scalar.
7. Avoid broad posterior machinery, multi-starts, EP, full PyTorch INLA, or NPE-context
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

uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family n --methods current --max-datasets 1000 --batch-size 32 \
    --normal-eb-sigma-grid-refine --normal-beta-sigma-grid

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-mixed,medium-n-mixed,large-n-mixed,huge-n-mixed \
    --partition train --n-epochs 2 --n-inla 1000 --n-total 1000 \
    --analytical-methods normal_eb,normal_sigma_grid --re-correlation diagonal

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-sampled,medium-n-sampled,large-n-sampled,huge-n-sampled \
    --partition valid --n-inla 1000 --n-total 1000 \
    --analytical-methods normal_eb,normal_sigma_grid --re-correlation diagonal

uv run python -u experiments/analytical/glmm_normal_inla_diagnostic.py \
    --data-ids huge-n-sampled --partition valid --tail-scan 8000 --tail-k 16 \
    --tail-metric blup_eb_rmse --methods normal_sigma_grid_srfx --batch-size 32

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
