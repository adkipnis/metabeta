Poisson GLMM Plan
=================

Last updated: 2026-05-22 (Poisson efficiency pass)

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

The final VG-centered σ averaging stage is no longer part of the default. It is retained as
an explicit ablation because it was FFX-neutral on all first-1000 rows and cost roughly
`2-7 ms/ds`.

Useful benchmark methods:

- `current` / `default`: full retained path.
- `raw`: PQL/raw baseline.
- `poisson_eb`: diagonal Laplace EB only.
- `poisson_marginal_beta`: EB plus marginal-mean β correction.
- `poisson_laplace_pirls_diag`: EB plus diagonal joint PIRLS.
- `poisson_laplace_pirls_sigma_grid`: default path through σ grid.
- `poisson_laplace_pirls_sigma_avg`: default path through pre-VG σ averaging.
- `poisson_variational_gaussian`: default initializer plus VG refinement; matches current.
- `poisson_variational_gaussian_sigma_avg`: VG plus scalar σ averaging ablation.

First-1000 INLA Comparison
--------------------------

Sequential CPU rerun on 2026-05-22. Lower NRMSE is better. INLA values are first-1000
diagonal R-INLA references.

| Dataset | part | current FFX | INLA FFX | gap | current σ | INLA σ | current BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2077 | 0.1835 | +0.0242 | 0.4268 | 0.3404 | 0.5208 | 0.4936 | 48.7 |
| small-p-sampled | valid | 0.2338 | 0.2276 | +0.0062 | 0.4783 | 0.4356 | 0.5327 | 0.5309 | 43.3 |
| small-p-sampled | test | 0.2112 | 0.1997 | +0.0115 | 0.4388 | 0.3966 | 0.5211 | 0.5281 | 44.7 |
| medium-p-mixed | train | 0.1807 | 0.1675 | +0.0132 | 0.4003 | 0.3214 | 0.5075 | 0.4789 | 83.7 |
| medium-p-sampled | valid | 0.2310 | 0.2146 | +0.0164 | 0.4556 | 0.4209 | 0.5504 | 0.5618 | 88.3 |
| medium-p-sampled | test | 0.2385 | 0.2267 | +0.0118 | 0.4354 | 0.3883 | 0.5504 | 0.5849 | 83.3 |
| large-p-mixed | train | 0.1936 | 0.1778 | +0.0158 | 0.4239 | 0.3076 | 0.5410 | 0.5001 | 96.1 |
| large-p-sampled | valid | 0.2553 | 0.2467 | +0.0086 | 0.4999 | 0.4232 | 0.6132 | 0.5870 | 97.3 |
| large-p-sampled | test | 0.2280 | 0.2186 | +0.0094 | 0.4286 | 0.3439 | 0.5491 | 0.5618 | 107.4 |
| huge-p-mixed | train | 0.2028 | 0.2080 | -0.0052 | 0.4591 | 0.3428 | 0.5717 | 0.5453 | 122.5 |
| huge-p-sampled | valid | 0.2650 | 0.2582 | +0.0068 | 0.5956 | 0.3944 | 0.6223 | 0.6223 | 156.0 |
| huge-p-sampled | test | 0.2555 | 0.2238 | +0.0317 | 0.5203 | 0.3560 | 0.6171 | 0.5956 | 144.5 |

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
epochs and sampled valid/test. These were run before dropping final VG σ averaging from
the default; FFX/sigma/BLUP should be effectively unchanged, with expected runtime lower
by about `2-7 ms/ds`.

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

Current warm stage timings after the VG fusion patch (2026-05-23):

| stage | small-p-mixed first-1000 ms/ds |
| --- | ---: |
| EB | ~25 |
| PIRLS | ~1.7 |
| marginal β | ~0.3 |
| σ grid | ~1.6 |
| pre-VG σ averaging | ~2.2 |
| VG | ~11 (was ~14 before fusion) |
| final VG σ averaging | retired from default |

Accepted speed cleanups:

- Final VG σ averaging refresh steps reduced from `2` to `1`. This saved roughly
  `7%` overall on the first-1000 table while preserving FFX.
- Final VG σ averaging was then removed from the default entirely. Across all 12
  first-1000 rows, dropping it changed FFX by at most about `0.0008` and saved roughly
  `2-7 ms/ds`.
- VG adaptive continuation reduced to one target-accepted step.
- Duplicate all-ones σ candidates are no longer evaluated.
- VG offset computation now reuses already masked `Z` inside retained VG steps.
- **VG step+covariance fusion** (`_poissonVariationalStepAndCovDiag`): inner loop now
  fuses `_poissonVariationalStepDiag` + `_poissonVariationalCovarianceDiag` into one
  call, sharing `Z_eff`, the variational offset, and the precision diagonal. Eliminates
  one full Hessian evaluation per inner step (25 inner + 2 final + 1 adaptive = ~28
  fewer per VG call). Reduces VG from ~14 → ~11 ms/ds on small-p-mixed (~21% VG
  reduction). FFX change ≤ 0.0001 across all sizes. Post-sigma-update covariance
  refreshes are kept as-is since σ changed.

Latency mode (`poisson_latency` benchmark method, 2026-05-23):

- Registered `poisson_latency` as a named benchmark method: identical pipeline to
  `current` (EB+PIRLS+marginal β+σ grid+VG) but with d-gated EB `fast_steps=8,
  fast_max_d=8`. Wired through `_poissonEbKwargs`.

First-1000 latency-mode comparison (2026-05-23):

| Dataset | part | current FFX | latency FFX | Δ FFX | current ms/ds | latency ms/ds | Δ ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2077 | 0.2076 | -0.0001 | 49.4 | 33.9 | -31% |
| small-p-sampled | valid | 0.2338 | 0.2326 | -0.0012 | 45.2 | 32.9 | -27% |
| small-p-sampled | test | 0.2112 | 0.2120 | +0.0008 | 44.1 | 31.9 | -28% |
| medium-p-mixed | train | 0.1807 | 0.1800 | -0.0007 | 78.8 | 58.6 | -26% |
| medium-p-sampled | valid | 0.2310 | 0.2307 | -0.0003 | 82.1 | 61.3 | -25% |
| medium-p-sampled | test | 0.2385 | 0.2386 | +0.0001 | 78.0 | 59.2 | -24% |
| large-p-mixed | train | 0.1936 | 0.1936 | 0.0000 | 89.2 | 89.3 | ~0% |
| large-p-sampled | valid | 0.2553 | 0.2553 | 0.0000 | 90.8 | 91.0 | ~0% |
| large-p-sampled | test | 0.2280 | 0.2280 | 0.0000 | 102.0 | 102.1 | ~0% |
| huge-p-mixed | train | 0.2028 | 0.2028 | 0.0000 | 130.2 | 129.2 | ~0% |
| huge-p-sampled | valid | 0.2650 | 0.2650 | 0.0000 | 164.3 | 164.4 | ~0% |
| huge-p-sampled | test | 0.2555 | 0.2555 | 0.0000 | 147.0 | 146.0 | ~0% |

Large/huge are identical because their `d > 8` so the full EB budget runs. Latency σ/BLUP
slightly worse on small/medium only (Δσ ≤ 0.022, ΔBLUP ≤ 0.003). No FFX regression on
any size at 4 decimal precision.

Optional speed mode:

- `--poisson-eb-steps 8` cuts roughly `10-35 ms/ds` and preserves FFX on tested first-1000
  rows, but materially regresses σ/BLUP on huge sampled rows. Keep it opt-in, not default.
- `--poisson-eb-fast-steps 8 --poisson-eb-fast-max-d 8` (the `poisson_latency` method)
  applies the shortcut only to small/medium rows (`d <= 8`), leaving large/huge on the full
  EB budget. Saves 24-31% on small/medium with no FFX regression. This is now the
  registered latency mode.
- Reduced VG budget (`outer=4, inner=4, final=1`) is not a default candidate: it is nearly
  neutral on small/medium but materially regresses huge sampled rows.

Retired Directions
------------------

These are summarized here only to avoid repeating them:

- Full-Σ joint PIRLS was stable but worse than the diagonal default.
- Direct VG from PQL was faster but materially less accurate than using the EB/PIRLS/grid
  initializer.
- Wider scalar σ grids, true-σ and INLA-σ plug-ins, and β-only AGQ/grid variants did not
  close the FFX gap.
- Local β skew correction worsened FFX on representative rows.
- Multi-state VG averaging, final VG σ averaging, and VG line-search polish were
  FFX-neutral or too fragile for default use. Final VG σ averaging remains as an explicit
  benchmark ablation; the other two were removed from the live method surface.
- Further gate tuning around the old EB/grid path has low expected value.

Next Directions
---------------

1. ~~**Low-level retained-stage profiling.**~~ **Done (2026-05-23).**
   VG: fused `_poissonVariationalStepAndCovDiag` eliminates ~28 redundant Hessian
   evaluations per VG call by sharing `Z_eff`, offset, and precision diagonal. Saves
   ~3 ms/ds on small (~21% VG reduction); EB inner loop has no equivalent shortcut
   (post-loop H recomputation is necessary because H is needed at the final IRLS state).
   Further EB speedup would require reducing `n_inner` or `n_steps`, both of which
   regress σ/BLUP.

2. ~~**Explicit latency/accuracy modes.**~~ **Done (2026-05-23).**
   `poisson_latency` benchmark method registered: full current pipeline with d-gated EB
   (`fast_steps=8, fast_max_d=8`). Saves 24-31% on small/medium, zero impact on
   large/huge. No FFX regression at any size. See latency-mode table above.

3. **Architecture only if FFX becomes limiting again.**
   The remaining INLA FFX gap appears more like posterior-mean/target mismatch than σ
   scale or covariance shape. The next serious diagnostic would compare INLA β deltas
   against our VG/Laplace target on the hardest rows before adding new estimators.

Commands
--------

```bash
# Accuracy + stage timing (one size at a time for clean walltime):
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_variational_gaussian poisson_variational_gaussian_sigma_avg \
    --sizes small --batch-size 32 --max-datasets 1000 --poisson-stage-timings

# Latency-mode comparison (one size at a time):
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_latency \
    --sizes small --batch-size 32 --max-datasets 1000 --poisson-stage-timings

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/glmm experiments/analytical
```
