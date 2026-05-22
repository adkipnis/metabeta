Poisson GLMM Plan
=================

Last updated: 2026-05-22 (Poisson cleanup pass)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA remains
the reference target only; R-INLA as a backend and full PyTorch-INLA are out of scope.

Current Default
---------------

The retained Poisson path is:

1. RAW/PQL Poisson GLMM initialization.
2. Diagonal single-mode Laplace EB over β and diagonal σ_rfx.
3. Fixed-budget diagonal-Σ joint Laplace-PIRLS over β/u/σ.
4. Marginal-mean β correction.
5. Conservative full-candidate diagonal σ grid with scales `(0.5, 0.75, 1.0)`.
6. Scalar Laplace-weighted σ averaging with local fixed-σ β/u PIRLS refresh.
7. Variational-Gaussian posterior-mean refinement with
   `outer=5, inner=5, final=2, damping=0.7`, plus one adaptive continuation step.
8. VG-centered scalar σ averaging with β-only weighted output and one refresh step.

Useful benchmark methods:

- `current` / `default`: full retained path.
- `raw`: PQL/raw baseline.
- `poisson_eb`: diagonal Laplace EB only.
- `poisson_marginal_beta`: EB plus marginal-mean β correction.
- `poisson_laplace_pirls_diag`: EB plus diagonal joint PIRLS.
- `poisson_laplace_pirls_sigma_grid`: default path through σ grid.
- `poisson_laplace_pirls_sigma_avg`: default path through pre-VG σ averaging.
- `poisson_variational_gaussian`: default initializer plus VG refinement.
- `poisson_variational_gaussian_sigma_avg`: VG plus scalar σ averaging; matches current.

First-1000 INLA Comparison
--------------------------

Sequential CPU rerun on 2026-05-22. Lower NRMSE is better. INLA values are first-1000
diagonal R-INLA references.

| Dataset | part | current FFX | INLA FFX | gap | current σ | INLA σ | current BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2077 | 0.1835 | +0.0242 | 0.4268 | 0.3404 | 0.5208 | 0.4936 | 55.6 |
| small-p-sampled | valid | 0.2330 | 0.2276 | +0.0054 | 0.4783 | 0.4356 | 0.5327 | 0.5309 | 50.6 |
| small-p-sampled | test | 0.2108 | 0.1997 | +0.0111 | 0.4388 | 0.3966 | 0.5211 | 0.5281 | 51.2 |
| medium-p-mixed | train | 0.1800 | 0.1675 | +0.0125 | 0.4003 | 0.3214 | 0.5075 | 0.4789 | 93.9 |
| medium-p-sampled | valid | 0.2305 | 0.2146 | +0.0159 | 0.4556 | 0.4209 | 0.5504 | 0.5618 | 97.6 |
| medium-p-sampled | test | 0.2384 | 0.2267 | +0.0117 | 0.4354 | 0.3883 | 0.5504 | 0.5849 | 94.7 |
| large-p-mixed | train | 0.1935 | 0.1778 | +0.0157 | 0.4239 | 0.3076 | 0.5410 | 0.5001 | 106.7 |
| large-p-sampled | valid | 0.2552 | 0.2467 | +0.0085 | 0.4999 | 0.4232 | 0.6132 | 0.5870 | 108.3 |
| large-p-sampled | test | 0.2279 | 0.2186 | +0.0093 | 0.4286 | 0.3439 | 0.5491 | 0.5618 | 116.7 |
| huge-p-mixed | train | 0.2026 | 0.2080 | -0.0054 | 0.4591 | 0.3428 | 0.5717 | 0.5453 | 128.6 |
| huge-p-sampled | valid | 0.2648 | 0.2582 | +0.0066 | 0.5956 | 0.3944 | 0.6223 | 0.6223 | 165.6 |
| huge-p-sampled | test | 0.2553 | 0.2238 | +0.0315 | 0.5203 | 0.3560 | 0.6171 | 0.5956 | 152.8 |

Interpretation:

- FFX is now close to INLA on most first-1000 rows. The largest remaining gap is
  `huge-p-sampled:test` (`+0.0315`), followed by `small-p-mixed:train` (`+0.0242`).
- σ remains consistently worse than INLA, especially sampled rows, but true/INLA-σ
  plug-in diagnostics did not close the FFX gap. Do not prioritize σ-only patches unless
  downstream NPE context needs them.
- BLUP is mixed: current ties INLA on `huge-p-sampled:valid`, lags on mixed rows, and is
  close on several sampled rows.

Full 8192-Row Current Benchmark
-------------------------------

Full available Poisson rows per cell (`8192` indices), mixed train over two training
epochs and sampled valid/test. These are current-method results only.

| Dataset | part | N | FFX | σ | BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 8192 | 0.1668 | 0.3592 | 0.5104 | 74.3 |
| small-p-sampled | valid | 8192 | 0.2024 | 0.4225 | 0.5149 | 71.0 |
| small-p-sampled | test | 8192 | 0.1970 | 0.4195 | 0.5169 | 72.7 |
| medium-p-mixed | train | 8192 | 0.1706 | 0.4184 | 0.4967 | 128.2 |
| medium-p-sampled | valid | 8192 | 0.2078 | 0.4436 | 0.5516 | 127.2 |
| medium-p-sampled | test | 8192 | 0.2062 | 0.4517 | 0.5631 | 132.9 |
| large-p-mixed | train | 8192 | 0.2076 | 0.5085 | 0.7335 | 152.5 |
| large-p-sampled | valid | 8192 | 0.2224 | 0.5151 | 0.6077 | 148.0 |
| large-p-sampled | test | 8192 | 0.2254 | 0.5228 | 0.6482 | 160.0 |
| huge-p-mixed | train | 8192 | 0.2109 | 0.5435 | 0.5904 | 204.6 |
| huge-p-sampled | valid | 8192 | 0.2387 | 0.5910 | 0.6774 | 226.8 |
| huge-p-sampled | test | 8192 | 0.2650 | 0.5890 | 0.7061 | 208.0 |

Efficiency State
----------------

Warm stage timings show the retained hot stages are EB and VG:

| stage | small-p-mixed first-1000 ms/ds |
| --- | ---: |
| EB | ~25 |
| PIRLS | ~1.7 |
| marginal β | ~0.3 |
| σ grid | ~1.6 |
| pre-VG σ averaging | ~2.2 |
| VG | ~14 |
| final VG σ averaging | ~2.7 |

Accepted speed cleanups:

- Final VG σ averaging refresh steps reduced from `2` to `1`. This saved roughly
  `7%` overall on the first-1000 table while preserving FFX.
- VG adaptive continuation reduced to one target-accepted step.
- Duplicate all-ones σ candidates are no longer evaluated.
- VG offset computation now reuses already masked `Z` inside retained VG steps.

Optional speed mode:

- `--poisson-eb-steps 8` cuts roughly `10-35 ms/ds` and preserves FFX on tested first-1000
  rows, but materially regresses σ/BLUP on huge sampled rows. Keep it opt-in, not default.

Retired Directions
------------------

These are summarized here only to avoid repeating them:

- Full-Σ joint PIRLS was stable but worse than the diagonal default.
- Direct VG from PQL was faster but materially less accurate than using the EB/PIRLS/grid
  initializer.
- Wider scalar σ grids, true-σ and INLA-σ plug-ins, and β-only AGQ/grid variants did not
  close the FFX gap.
- Local β skew correction worsened FFX on representative rows.
- Multi-state VG averaging and VG line-search polish were FFX-neutral and have been
  removed from the live Poisson path and benchmark method surface.
- Further gate tuning around the old EB/grid path has low expected value.

Next Directions
---------------

1. **Low-level retained-stage profiling.**
   If more speed is needed, profile inside `refinePoissonLaplaceEb` and
   `refinePoissonVariationalGaussian`. Focus on repeated target/Hessian construction,
   Cholesky solves, and duplicate `mu`/offset recomputation. Avoid optimizing retired
   diagnostic branches.

2. **Explicit latency/accuracy modes.**
   Keep the current default as the accuracy mode. Add a documented FFX-first latency mode
   only if downstream evaluation tolerates the σ/BLUP regression from fewer EB steps.

3. **Architecture only if FFX becomes limiting again.**
   The remaining INLA FFX gap appears more like posterior-mean/target mismatch than σ
   scale or covariance shape. The next serious diagnostic would compare INLA β deltas
   against our VG/Laplace target on the hardest rows before adding new estimators.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_variational_gaussian poisson_variational_gaussian_sigma_avg \
    --sizes small medium large huge --batch-size 32 --max-datasets 1000 \
    --poisson-stage-timings

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/glmm experiments/analytical
```
