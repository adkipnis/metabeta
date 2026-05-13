Normal GLMM Plan
================

Last updated: 2026-05-13

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

3. **Monitor:** sigma(Eps) projection and final GLS scale. Projection change fixed
   large/huge outliers; oracle σ(Eps) does not improve point estimates under the
   current path.
4. **Secondary:** EM movement gates or damping. Only pursue if oracle attribution
   confirms EM is the limiting step in broad bins.

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
uv run python experiments/analytical/glmm_raw_diagnostic.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
