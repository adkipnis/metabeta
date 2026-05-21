Poisson GLMM Plan
=================

Last updated: 2026-05-21 (VG inner-budget default)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA is a
reference target only; using R-INLA as a backend and implementing full PyTorch-INLA remain
out of scope.

Current Default
---------------

The retained Poisson path is:

1. RAW/PQL Poisson GLMM initialization.
2. Diagonal single-mode Laplace EB over β and diagonal σ_rfx.
3. Fixed-budget diagonal-Σ joint Laplace-PIRLS over β/u/σ.
4. Marginal-mean β correction with `min_d=1`.
5. Conservative full-candidate diagonal σ grid with scales `(0.5, 0.75, 1.0)`.
6. Scalar Laplace-weighted σ averaging with local fixed-σ β/u PIRLS refresh.
7. Variational-Gaussian posterior-mean refinement with `outer=5, inner=5, final=2`.
8. VG-centered scalar σ averaging with β-only weighted output.

The final two stages are now the main useful architecture: they move β toward a
posterior-mean-like estimate by accounting for random-effect uncertainty in the Poisson
log link. The older EB/PIRLS/grid stages are still useful as a robust initializer; direct
VG from PQL was materially worse.

Useful benchmark methods:

- `current` / `default`: full retained path.
- `raw`: PQL/raw baseline.
- `poisson_eb`: diagonal Laplace EB only.
- `poisson_marginal_beta`: EB plus marginal-mean β correction.
- `poisson_laplace_pirls_diag`: EB plus diagonal joint PIRLS.
- `poisson_laplace_pirls_sigma_grid`: default path through the full-candidate σ grid.
- `poisson_laplace_pirls_sigma_avg`: default path through pre-VG σ averaging.
- `poisson_variational_gaussian`: default initializer plus VG posterior-mean refinement.
- `poisson_variational_gaussian_sigma_avg`: VG plus scalar σ averaging; matches current.

Current Evidence
----------------

First 1000 rows per selected cell, sequential CPU rerun on 2026-05-21 before the
`inner=5` default change. Lower NRMSE is better. INLA values are current first-1000
diagonal R-INLA references.

| Dataset | part | current FFX | INLA FFX | FFX gap | current σ | INLA σ | current BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2076 | 0.1835 | +0.0241 | 0.4272 | 0.3404 | 0.5205 | 0.4936 | 69.2 |
| small-p-sampled | valid | 0.2327 | 0.2276 | +0.0051 | 0.4784 | 0.4356 | 0.5325 | 0.5309 | 83.8 |
| medium-p-mixed | train | 0.1815 | 0.1675 | +0.0140 | 0.4003 | 0.3214 | 0.5072 | 0.4789 | 121.1 |
| medium-p-sampled | valid | 0.2363 | 0.2146 | +0.0217 | 0.4827 | 0.4209 | 0.5576 | 0.5618 | 124.7 |
| large-p-mixed | train | 0.1955 | 0.1778 | +0.0177 | 0.4365 | 0.3076 | 0.5425 | 0.5001 | 138.4 |
| large-p-sampled | valid | 0.2924 | 0.2467 | +0.0457 | 0.4990 | 0.4232 | 0.6204 | 0.5870 | 134.1 |

Interpretation:

- FFX is now close to INLA on small sampled and mixed medium/large rows.
- The main remaining FFX weakness is sampled large rows.
- σ remains consistently worse than INLA, but recent diagnostics show that simply plugging
  in true σ or INLA σ does not close the FFX gap.
- CPU runtime is above the original ideal on medium/large rows. Future patches need either
  a clear FFX gain or a cheaper approximation to the current VG behavior.

Retired Directions
------------------

These branches were removed from the live implementation or benchmark surface:

- Full-Σ joint PIRLS: stable, but worse than the diagonal default on tested rows. The
  failure suggests covariance shape is not the dominant missing ingredient.
- Fixed-σ direct Laplace-target autograd polish: small-row gains were tiny, and
  medium/large rows regressed after the VG default.
- Direct VG from PQL: faster, but FFX was much worse than using the EB/PIRLS/grid
  initializer.
- β-only AGQ optimizer and old β-only σ grid variants: slower or superseded by
  full-candidate σ grid and weighted σ averaging.
- Wider scalar σ grids, true-σ plug-ins, and INLA-σ plug-ins: useful diagnostics, but not
  a productive FFX lever.
- Additional gate tuning around the old EB/grid path: low expected value compared with
  improving the posterior-mean/VG update itself.

Diagnostics So Far
------------------

The hard-row oracle run ranked difficult sampled rows and compared current VG σ averaging,
wider σ averaging, fixed true/INLA σ refreshes, and high-budget VG.

Key findings:

- Wider σ averaging did not improve FFX (`0.6285 -> 0.6302` on the selected hard rows).
- Fixed true σ and INLA σ refreshes were worse for FFX than the current VG state.
- High-budget VG was the only material hard-row lever (`0.6285 -> 0.5377`) and also
  improved σ/BLUP.
- Therefore the remaining gap is more likely β/m/V posterior-mean convergence or target
  calibration than covariance scale, covariance shape, or a small set of outlier rows.

VG budget follow-up:

- A hard-row budget ladder showed that `outer=7` and `inner=5` both recover most of the
  high-budget VG FFX gain (`0.5748 -> ~0.545` on the selected high-FFX rows).
- Broader first-1000 sampled checks favored `inner=5`: it improved medium sampled valid
  FFX (`0.2363 -> 0.2305`) and large sampled valid FFX (`0.2924 -> 0.2572`) while staying
  neutral on small mixed/sampled and medium sampled test.
- `outer=7` was cheaper in update count but regressed small mixed FFX
  (`0.2076 -> 0.2127`), so it is not a safe blanket default.
- The default VG inner budget is therefore now `5`. Timing from these runs should be
  rerun cleanly because a background process was active.

Next Directions
---------------

Primary:

- Re-run the current default with clean timing and update the evidence table for
  small/medium/large mixed and sampled rows.
- Diagnose why additional β/m inner steps help sampled high-d rows: compare β step norm,
  random-effect mean step norm, V offset change, σ movement, and variational target change
  between `inner=3` and `inner=5`.
- Test VG target calibration variants with the row set fixed: prior strength, variance
  offset clipping, damping schedule, and acceptance by the variational target.

Secondary:

- Keep scalar/intercept-slope σ averaging as a low-cost ablation through the existing
  `scale_mode` options, but do not expand grids unless a new VG variant starts trading FFX
  gains against severe σ/BLUP regressions.
- Revisit full covariance only if a future diagnostic shows correlation-specific failures;
  current evidence does not justify carrying the full-Σ branch in production code.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_variational_gaussian poisson_variational_gaussian_sigma_avg \
    --sizes small medium large \
    --batch-size 32 --max-datasets 1000

uv run python -u experiments/analytical/glmm_poisson_hard_row_diagnostic.py \
    --combos medium-p-sampled:valid large-p-sampled:test \
    --max-datasets 1000 --top-k 64 --batch-size 32 --rank-by vg_rmse

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
