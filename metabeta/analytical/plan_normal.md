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

2. **Implement a normal-specific diagonal Laplace-EB calibration candidate.**
   Gaussian random-effect integration is analytically exact, so this should not copy the
   Bernoulli nested mode optimizer. The candidate should optimize only diagonal
   `sigma_rfx` and optionally `sigma_eps`, keep β fixed or update it through the existing
   final GLS recompute, and then reuse `_recomputeNormalFinalDiagMap`.

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

- Add an experiment-only refinement, e.g. `normal_laplace_eb`, gated behind a kwarg.
- Start from raw or current MAP diagonal `sigma_rfx`.
- Use the exact Gaussian marginal likelihood with diagonal Ψ and existing priors.
- Update log variances with a small fixed number of damped Newton/Fisher or coordinate
  steps. Keep β out of the optimizer; recompute β/BLUP once at the end through the
  existing diagonal final pass.
- Include objective acceptance against current MAP and finite-value fallback to the
  incoming stats.

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
uv run python experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-mixed,medium-n-mixed,large-n-mixed,huge-n-mixed \
    --partition train --n-epochs 2 --n-inla 1000 --n-total 1000 \
    --analytical-methods raw,map --normal-re-correlation diagonal
uv run python experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-sampled,medium-n-sampled,large-n-sampled,huge-n-sampled \
    --partition valid --n-inla 1000 --n-total 1000 \
    --analytical-methods raw,map --normal-re-correlation diagonal
uv run python experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-n-sampled,medium-n-sampled,large-n-sampled,huge-n-sampled \
    --partition test --n-inla 1000 --n-total 1000 \
    --analytical-methods raw,map --normal-re-correlation diagonal
uv run python experiments/analytical/glmm_raw_diagnostic.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
