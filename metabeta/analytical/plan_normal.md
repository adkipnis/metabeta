Normal GLMM Plan
================

Last updated: 2026-05-17

Production Path
---------------

`glmm()` with MAP σ_rfx refinement and diagonal-MAP final GLS/BLUP recompute.
MAP reports MAP `sigma_rfx_est`, writes a diagonal Ψ, and recomputes final
beta/BLUPs with that covariance. Off-diagonal correlations are excluded from the
final BLUP pass (estimated correlations are often harmful).

Baseline references:
- Current: `glmm()` (map_refine=True, map_recompute_blup=True)
- Raw: `glmm(..., map_refine=False)`
- Legacy (output-local MAP, no BLUP recompute): `glmm(..., map_recompute_blup=False)`

Oracle attribution takeaway (required-suite, `glmm_raw_diagnostic.py`): BLUP
ceiling is primarily Ψ/shrinkage. True Ψ diagonal → BLUP 0.4408 vs current 0.4682;
most of the MAP gain comes from the diagonal scale, not off-diagonals. True σ(Eps)
and oracle beta do not improve point estimates under the current recompute path.
Full tables in `glmm_perf_baseline.md`.

Closed Decisions
----------------

All details and cell-level tables are in `glmm_perf_baseline.md`.

- **REML:** all variants worse or catastrophically regressive on medium-n-mixed.
  Retired from package surface.
- **MAP ablation:** three-parameter joint optimization is Pareto-dominant over all
  subsets. `map_optimize` kwarg retained for diagnostics.
- **Beta blend:** d≤8→0.65, d>8→0.75 is optimal; oracle beta is globally worse.
  `beta_alpha_low`/`beta_alpha_high` kwargs retained.
- **BLUP variance calibration:** fixed in `_recomputeNormalFinalDiagMap`; 12/15
  n_g bins within ±10% of 1.0. Additive Ψ/G_mom floor dropped under MAP.

Open Priorities
---------------

1. **Run normal R-INLA reference comparisons.** Use `glmm_inla_comparison.py` on
   the normal mixed/train and sampled valid/test sets with `--analytical-methods raw,map`
   and `--normal-re-correlation diagonal`. The exact correlated Gaussian INLA path is
   numerically unstable on these data; diagonal INLA matches the production final
   covariance assumption and gives a useful reference for variance-scale calibration.

2. **Validate the normal-specific diagonal Laplace-EB calibration candidate.**
   Gaussian random-effect integration is analytically exact, so this should not copy the
   Bernoulli nested mode optimizer. The current prototype is `normal_laplace_eb=True`:
   a posterior-moment diagonal σ update with prior pseudo-weight `4`, objective acceptance,
   and the existing `_recomputeNormalFinalDiagMap` pass.

3. **Promote only if it is simpler or faster than current MAP, or closes a measured
   INLA gap.** Current MAP is an Adam loop over marginal Gaussian likelihood parameters.
   A worthwhile EB path should be closed-form, Newton/Fisher scoring, or 1-3 damped
   diagonal updates; another long optimizer is not worth adding.

4. **Monitor:** sigma(Eps) projection and final GLS scale. Projection change fixed
   large/huge outliers; oracle σ(Eps) does not improve point estimates under the
   current path.
5. **Secondary:** EM movement gates or damping. Only pursue if oracle attribution
   confirms EM is the limiting step in broad bins.

Laplace-EB Candidate Plan
-------------------------

**N1. Reference and diagnostics.**

- Finish the 1k-per-row R-INLA comparison for mixed/train and sampled valid/test.
- Record raw, MAP, diagonal R-INLA accuracy, and wall time per method.
- Use the first two completed mixed rows as an early signal: diagonal R-INLA improves
  `sigma_rfx` and BLUP modestly over MAP, and can improve FFX strongly on
  `medium-n-mixed`, but is about seconds per dataset versus milliseconds for MAP.

**N2. Fast diagonal EB update.**

- Current prototype is implemented as `normal_laplace_eb=True`, with benchmark method
  `normal_eb`.
- Start from current MAP diagonal `sigma_rfx`.
- Use the exact Gaussian marginal likelihood with diagonal Ψ and existing priors.
- Use posterior BLUP moments `E[b_j^2] ≈ bhat_j^2 + Var(b_j)`, blend with the prior
  scale as a pseudo-count, and accept only finite objective-nondecreasing updates.
- Keep β out of the optimizer; recompute β/BLUP once at the end through the existing
  diagonal final pass.

**N3. Speed-first approximation.**

- If N2 is accurate but not faster than MAP, derive a one-shot empirical-Bayes shrinkage
  correction from posterior BLUP moments: roughly `E[b_j^2] = bhat_j^2 + Var(b_j)`,
  aggregated across active groups, then blended with the prior scale.
- This is the likely production candidate if it matches most of the INLA σ/BLUP gain
  while staying close to raw/GLS runtime.

**N4. Promotion gate.**

- Benchmark `raw`, `map`, and `normal_laplace_eb` on the required 12-row normal suite.
- Promote only if global BLUP improves or wall time drops materially with no meaningful
  FFX/σ(Eps) regression.
- Retire the branch if it only improves σ_rfx without moving BLUP, or if it adds another
  optimizer with MAP-like runtime and similar accuracy.

Prototype Snapshot
------------------

First 1000 datasets per row, current MAP vs `normal_eb`. Lower NRMSE is better.

| Dataset | part | MAP FFX | EB FFX | MAP σ | EB σ | MAP σ_eps | EB σ_eps | MAP BLUP | EB BLUP | MAP ms | EB ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1096 | 0.1095 | 0.4814 | 0.4203 | 0.2151 | 0.2151 | 0.4192 | 0.4173 | 3.09 | 2.53 |
| small-n-sampled | valid | 0.2587 | 0.2588 | 0.5654 | 0.5646 | 0.2169 | 0.2169 | 0.5130 | 0.5125 | 2.28 | 2.53 |
| small-n-sampled | test | 0.2828 | 0.2827 | 0.4864 | 0.4684 | 0.2169 | 0.2169 | 0.4926 | 0.4924 | 2.25 | 2.50 |
| medium-n-mixed | train | 0.5489 | 0.5486 | 0.3798 | 0.3628 | 0.1655 | 0.1655 | 0.4409 | 0.4395 | 3.06 | 3.43 |
| medium-n-sampled | valid | 0.5305 | 0.5285 | 0.5709 | 0.4142 | 0.1891 | 0.1891 | 0.5426 | 0.5417 | 3.76 | 4.21 |
| medium-n-sampled | test | 0.4155 | 0.4127 | 0.4505 | 0.3980 | 0.1949 | 0.1949 | 0.4668 | 0.4640 | 3.69 | 4.05 |
| large-n-mixed | train | 1.8207 | 1.8169 | 0.4148 | 0.3718 | 0.1268 | 0.1268 | 0.4361 | 0.4359 | 4.14 | 4.70 |
| large-n-sampled | valid | 1.0820 | 1.0813 | 0.4982 | 0.4321 | 0.1563 | 0.1563 | 0.5347 | 0.5290 | 4.34 | 4.87 |
| large-n-sampled | test | 0.8186 | 0.7408 | 0.8721 | 0.4421 | 0.1513 | 0.1513 | 0.5261 | 0.5224 | 4.60 | 5.14 |
| huge-n-mixed | train | 1.3100 | 0.9413 | 0.4280 | 0.3782 | 0.1161 | 0.1161 | 0.4742 | 0.4716 | 5.24 | 5.90 |
| huge-n-sampled | valid | 1.0284 | 1.0359 | 0.4100 | 0.3698 | 0.1375 | 0.1375 | 0.4836 | 0.4818 | 6.19 | 7.07 |
| huge-n-sampled | test | 0.8872 | 0.8953 | 0.4092 | 0.3850 | 0.1438 | 0.1438 | 0.4823 | 0.4813 | 6.46 | 7.31 |

Takeaway: the prototype improves σ_rfx and BLUP in every row while keeping σ_eps fixed.
FFX is mostly neutral, with meaningful improvements on large/huge mixed/test and small
regressions on huge sampled rows. Runtime remains single-digit milliseconds per dataset.

Diagonal R-INLA Snapshot
-----------------------

Mixed/train rows, first 1000 datasets per row. This run used `raw,map` only; it completed
before `normal_eb` was added to `glmm_inla_comparison.py`. The INLA reference uses diagonal
random effects because the exact correlated Gaussian INLA branch was numerically unstable on
these datasets.

| Dataset | RAW FFX | MAP FFX | INLA FFX | RAW σ | MAP σ | INLA σ | RAW BLUP | MAP BLUP | INLA BLUP | RAW ms | MAP ms | INLA s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-n-mixed | 0.1124 | 0.1096 | 0.0985 | 0.7978 | 0.4814 | 0.3665 | 0.4761 | 0.4192 | 0.4081 | 0.37 | 7.26 | 2.368 |
| medium-n-mixed | 0.5758 | 0.5489 | 0.2301 | 0.5236 | 0.3798 | 0.3421 | 0.4550 | 0.4409 | 0.4288 | 0.57 | 2.81 | 2.614 |
| large-n-mixed | 1.7363 | 1.8207 | 0.2377 | 0.5449 | 0.4148 | 0.3397 | 0.4675 | 0.4361 | 0.4185 | 0.85 | 3.80 | 2.786 |
| huge-n-mixed | 1.0635 | 1.3100 | 0.2413 | 0.5752 | 0.4280 | 0.2809 | 0.4925 | 0.4742 | 0.4548 | 1.22 | 5.11 | 3.071 |

Takeaway: diagonal INLA is the accuracy leader on mixed/train, especially FFX for
medium/large/huge. MAP closes much of the raw σ/BLUP gap but still leaves about
`0.012-0.019` BLUP NRMSE on medium/large/huge and a larger σ gap on huge. INLA is
roughly seconds per dataset versus low single-digit milliseconds for analytical MAP.

Acceptance Criteria
-------------------

A raw-estimator change must improve at least one primary output on the required
suite (small/medium/large/huge × mixed-train + sampled-valid/test) without
material regressions:

- FFX, σ(Eps), and BLUP improvements outweigh σ(RFX) alone.
- σ(RFX) raw improvement is still valuable if it improves BLUP or reduces MAP
  dependency.
- Compare against both `raw` and `current` production MAP.
- Narrow-bin-only improvements stay as experiments.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_required_benchmark.py --methods current raw
uv run python experiments/analytical/glmm_required_benchmark.py \
    --family n --methods current normal_eb --max-datasets 1000 --batch-size 32
uv run python experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-mixed,medium-n-mixed,large-n-mixed,huge-n-mixed \
    --partition train --n-epochs 2 --n-inla 1000 --n-total 1000 \
    --analytical-methods raw,map,normal_eb --normal-re-correlation diagonal
uv run python experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-sampled,medium-n-sampled,large-n-sampled,huge-n-sampled \
    --partition valid --n-inla 1000 --n-total 1000 \
    --analytical-methods raw,map,normal_eb --normal-re-correlation diagonal
uv run python experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-sampled,medium-n-sampled,large-n-sampled,huge-n-sampled \
    --partition test --n-inla 1000 --n-total 1000 \
    --analytical-methods raw,map,normal_eb --normal-re-correlation diagonal
uv run python experiments/analytical/glmm_raw_diagnostic.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
