Poisson GLMM Plan
=================

Last updated: 2026-05-21 (Poisson next-step re-evaluation)

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

First 1000 rows per selected cell, sequential CPU rerun on 2026-05-21. Lower NRMSE is
better. INLA values are current first-1000 diagonal R-INLA references. The first table is
the last full INLA comparison snapshot before the `inner=5` VG default change; the second
table records the measured `inner=5` deltas that are now part of the default path. The
timings from these runs are considered unbiased.

| Dataset | part | current FFX | INLA FFX | FFX gap | current σ | INLA σ | current BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2076 | 0.1835 | +0.0241 | 0.4272 | 0.3404 | 0.5205 | 0.4936 | 69.2 |
| small-p-sampled | valid | 0.2327 | 0.2276 | +0.0051 | 0.4784 | 0.4356 | 0.5325 | 0.5309 | 83.8 |
| medium-p-mixed | train | 0.1815 | 0.1675 | +0.0140 | 0.4003 | 0.3214 | 0.5072 | 0.4789 | 121.1 |
| medium-p-sampled | valid | 0.2363 | 0.2146 | +0.0217 | 0.4827 | 0.4209 | 0.5576 | 0.5618 | 124.7 |
| large-p-mixed | train | 0.1955 | 0.1778 | +0.0177 | 0.4365 | 0.3076 | 0.5425 | 0.5001 | 138.4 |
| large-p-sampled | valid | 0.2924 | 0.2467 | +0.0457 | 0.4990 | 0.4232 | 0.6204 | 0.5870 | 134.1 |

Known `inner=5` default deltas:

| Dataset | part | old FFX | inner=5 FFX | INLA FFX | new FFX gap | σ change | BLUP change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2076 | 0.2076 | 0.1835 | +0.0241 | 0.4272 -> 0.4271 | 0.5205 -> 0.5206 |
| small-p-sampled | valid | 0.2327 | 0.2328 | 0.2276 | +0.0052 | flat | flat |
| medium-p-sampled | valid | 0.2363 | 0.2305 | 0.2146 | +0.0159 | 0.4827 -> 0.4619 | 0.5576 -> 0.5508 |
| medium-p-sampled | test | 0.2383 | 0.2383 | TBD | TBD | 0.4442 -> 0.4440 | 0.5509 -> 0.5508 |
| large-p-sampled | valid | 0.2924 | 0.2572 | 0.2467 | +0.0105 | 0.4990 -> 0.5086 | 0.6204 -> 0.6176 |
| large-p-sampled | test | 0.2304 | 0.2279 | TBD | TBD | 0.4519 -> 0.4375 | 0.5552 -> 0.5499 |

Interpretation:

- FFX is now close to INLA on small sampled and large sampled valid after `inner=5`.
- The remaining FFX gaps are concentrated in mixed rows and medium sampled rows, with
  small mixed still roughly `+0.024` and medium sampled valid roughly `+0.016`.
- σ remains consistently worse than INLA, especially on mixed rows, but recent diagnostics
  show that simply plugging in true σ or INLA σ does not close the FFX gap. σ is therefore
  not the primary next optimization target unless it changes β/m posterior means.
- BLUP remains moderately behind INLA on most rows. The current VG update improves BLUP
  together with FFX on sampled rows, which points to posterior-mean state convergence
  rather than isolated σ correction.
- CPU runtime is acceptable for now. Future patches should prioritize FFX movement; speed
  work is secondary until the estimator is closer to INLA.

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
- The default VG inner budget is therefore now `5`.

Current diagnosis:

- The old remaining large-sampled FFX gap was mostly a VG convergence-budget issue, and
  `inner=5` captured a large part of it without a broad regression.
- The remaining FFX gap is probably not explained by diagonal σ scale, full covariance, or
  scalar hyperparameter averaging. Those were tested and either failed to move FFX or
  regressed stability.
- The highest-probability remaining issue is that the variational-Gaussian fixed point is
  still a biased approximation to INLA posterior means in some row types. The likely
  sources are target calibration, damping/acceptance, and missing posterior averaging of β
  itself rather than missing σ scale exploration.
- Hard-row improvements from extra VG budget suggest there is still useful signal in
  β/m/V updates. Patches should make that signal row-adaptive or better calibrated instead
  of blindly increasing global iteration count.

Next Directions
---------------

Primary patch candidates:

1. **Adaptive VG continuation.**
   Add a cheap diagnostic after the default VG pass: β step norm, random-effect mean step
   norm, V offset movement, σ movement, and variational target improvement between the last
   two inner steps. Run 1-2 extra inner blocks only for rows with unresolved movement. This
   directly targets the high-budget VG win while avoiding the small-mixed regression seen
   with blanket `outer=7`.

2. **VG target calibration.**
   With the diagnostic row set fixed, test prior strength, variance-offset clipping,
   damping schedule, and target-based acceptance. The goal is to reduce mixed-row FFX gaps
   where extra iteration alone is not enough. This is more plausible than more σ grids
   because true/INLA σ refreshes did not improve FFX.

3. **β posterior-mean correction on top of VG.**
   Prototype a small final correction that uses the VG/Laplace β curvature to move from a
   conditional/variational mode toward an approximate β posterior mean. INLA reports
   posterior means, while our path is still mostly a deterministic fixed point plus σ
   averaging. This is the most likely architectural patch if calibration does not close the
   small-mixed and medium-sampled FFX gaps.

4. **Row-type diagnostic before any wider architecture.**
   Compare remaining bad rows by `d`, `q`, random-intercept/slope composition, marginal
   variance correction `0.5 z'Vz`, VG effective candidate count, and final target
   residual. If bad rows are strongly structured, specialize the adaptive continuation or
   β-mean correction to those rows; otherwise treat the issue as a global approximation
   mismatch.

Secondary:

- Keep scalar/intercept-slope σ averaging as a low-cost ablation through the existing
  `scale_mode` options, but do not expand grids unless a new VG variant starts trading FFX
  gains against severe σ/BLUP regressions.
- Revisit full covariance only if a future diagnostic shows correlation-specific failures;
  current evidence does not justify carrying the full-Σ branch in production code.
- Do not spend more effort on old EB/grid gates unless they are needed as robust
  initializers for a better VG/posterior-mean patch.

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
