Poisson GLMM Plan
=================

Last updated: 2026-05-21 (adaptive VG default)

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
7. Variational-Gaussian posterior-mean refinement with
   `outer=5, inner=5, final=2, damping=0.7`, plus two adaptive continuation steps.
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

First 1000 rows per cell, sequential CPU rerun on 2026-05-21. Lower NRMSE is better.
INLA values are current first-1000 diagonal R-INLA references. Huge INLA fits are not
integrated yet, so those cells are deliberately left blank. Timings from these runs are
considered unbiased.

| Dataset | part | current FFX | INLA FFX | FFX gap | current σ | INLA σ | current BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2076 | 0.1835 | +0.0241 | 0.4260 | 0.3404 | 0.5204 | 0.4936 | 59.2 |
| small-p-sampled | valid | 0.2326 | 0.2276 | +0.0050 | 0.4779 | 0.4356 | 0.5324 | 0.5309 | 53.9 |
| small-p-sampled | test | 0.2107 | 0.1997 | +0.0110 | 0.4389 | 0.3966 | 0.5206 | 0.5281 | 55.1 |
| medium-p-mixed | train | 0.1807 | 0.1675 | +0.0132 | 0.3976 | 0.3214 | 0.5068 | 0.4789 | 102.9 |
| medium-p-sampled | valid | 0.2304 | 0.2146 | +0.0158 | 0.4500 | 0.4209 | 0.5497 | 0.5618 | 103.7 |
| medium-p-sampled | test | 0.2383 | 0.2267 | +0.0116 | 0.4388 | 0.3883 | 0.5504 | 0.5849 | 97.1 |
| large-p-mixed | train | 0.1935 | 0.1778 | +0.0157 | 0.4215 | 0.3076 | 0.5405 | 0.5001 | 111.3 |
| large-p-sampled | valid | 0.2551 | 0.2467 | +0.0084 | 0.4958 | 0.4232 | 0.6123 | 0.5870 | 112.9 |
| large-p-sampled | test | 0.2278 | 0.2186 | +0.0092 | 0.4249 | 0.3439 | 0.5485 | 0.5618 | 124.3 |
| huge-p-mixed | train | 0.2026 | — | — | 0.4549 | — | 0.5710 | — | 141.7 |
| huge-p-sampled | valid | 0.2646 | — | — | 0.5703 | — | 0.6195 | — | 181.0 |
| huge-p-sampled | test | 0.2549 | — | — | 0.5052 | — | 0.6154 | — | 168.8 |

Interpretation:

- FFX is now within about `+0.005` to `+0.016` of INLA on sampled small/medium/large rows
  and around `+0.013` to `+0.024` behind on mixed rows.
- σ remains consistently worse than INLA, especially on mixed rows, but recent diagnostics
  show that simply plugging in true σ or INLA σ does not close the FFX gap. σ is therefore
  not the primary next optimization target unless it changes β/m posterior means.
- BLUP is mixed: current is better than INLA on several sampled rows but remains behind on
  mixed rows and large sampled valid. Further BLUP work is secondary until FFX moves.
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
- Local β skew/posterior-mean correction after VG: tested both signs on representative
  rows; it worsened FFX (`small-p-mixed` 200-row `0.2857 -> 0.2887`, `medium-p-sampled`
  valid 200-row `0.2428 -> 0.2469`, `large-p-sampled` valid 200-row `0.2555 -> 0.2607`).
  A one-shot third-derivative β shift is therefore not the right approximation.
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

VG budget and calibration follow-up:

- A hard-row budget ladder showed that `outer=7` and `inner=5` both recover most of the
  high-budget VG FFX gain (`0.5748 -> ~0.545` on the selected high-FFX rows).
- Broader first-1000 sampled checks favored `inner=5`: it improved medium sampled valid
  FFX (`0.2363 -> 0.2305`) and large sampled valid FFX (`0.2924 -> 0.2572`) while staying
  neutral on small mixed/sampled and medium sampled test.
- `outer=7` was cheaper in update count but regressed small mixed FFX
  (`0.2076 -> 0.2127`), so it is not a safe blanket default.
- Adaptive VG continuation with two target-accepted extra steps was flat or better on all
  small/medium/large 1k rows and improved σ/BLUP on sampled rows. It is now default.
- VG damping `0.7` was flat or better than `0.5` on all small/medium/large 1k rows, with
  the clearest gains on large sampled valid (`0.2565 -> 0.2551`) and medium/large mixed.
  It is now default.
- VG prior strength (`2` or `8`) and offset clipping (`0.5`) did not materially improve
  FFX; keep them as opt-in diagnostics rather than defaults.

Current diagnosis:

- The old large-sampled FFX gap was mostly a VG convergence/calibration issue; `inner=5`,
  adaptive continuation, and damping `0.7` reduced valid from `0.2924` to `0.2551`.
- The remaining FFX gap is probably not explained by diagonal σ scale, full covariance, or
  scalar hyperparameter averaging. Those were tested and either failed to move FFX or
  regressed stability.
- The highest-probability remaining issue is that the variational-Gaussian fixed point is
  still a biased approximation to INLA posterior means in some row types. A local β skew
  shift failed, so the useful version probably needs to change the joint q(u)/β state or
  average over multiple nearby VG fixed points rather than perturb β alone.
- Hard-row improvements from extra VG budget suggest there is still useful signal in
  β/m/V updates. Patches should make that signal row-adaptive or better calibrated instead
  of blindly increasing global iteration count.

Next Directions
---------------

Primary patch candidates:

1. **Multi-start / multi-temperature VG posterior averaging.**
   Run a tiny set of VG continuations from deliberately perturbed β/m/V states or ELBO
   temperatures, then average β under the same variational target. The failed local β skew
   correction suggests that posterior-mean movement is not captured by a one-shot β-only
   formula; high-budget VG still improves hard rows, so averaging nearby fixed points is
   the next coherent approximation to INLA-style latent posterior averaging.

2. **Row-type diagnostic before any wider architecture.**
   Compare remaining bad rows by `d`, `q`, random-intercept/slope composition, marginal
   variance correction `0.5 z'Vz`, VG effective candidate count, and final target
   residual. If bad rows are strongly structured, specialize the adaptive continuation or
   multi-state averaging to those rows; otherwise treat the issue as a global approximation
   mismatch.

3. **Natural-gradient / damped ELBO polish for β and q(u).**
   Prototype one or two target-accepted updates that optimize the actual VG objective
   more directly than the current block Newton surrogate. This is heavier than the current
   path, but it targets the same failure mode as high-budget VG without more σ grids.

4. **Adaptive VG refinement cleanup.**
   Keep the adaptive continuation diagnostics (`accept_count`, β/m/offset/σ movement) and
   use them to decide whether future extra work should be row-gated more aggressively. Do
   not increase global VG iteration count unless a new diagnostic shows a broad unresolved
   movement pattern.

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
