Poisson GLMM Plan
=================

Last updated: 2026-05-22 (8192-row current benchmark added)

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
- `poisson_variational_gaussian_state_avg`: current plus multi-state VG averaging prototype.
- `poisson_variational_gaussian_polish`: current plus target-accepted VG polish prototype.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU rerun on 2026-05-21. Lower NRMSE is better.
INLA values are current first-1000 diagonal R-INLA references. Timings from these runs are
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
| huge-p-mixed | train | 0.2026 | 0.2080 | -0.0054 | 0.4549 | 0.3428 | 0.5710 | 0.5453 | 141.7 |
| huge-p-sampled | valid | 0.2646 | 0.2582 | +0.0064 | 0.5703 | 0.3944 | 0.6195 | 0.6223 | 181.0 |
| huge-p-sampled | test | 0.2549 | 0.2238 | +0.0311 | 0.5052 | 0.3560 | 0.6154 | 0.5956 | 168.8 |

Interpretation:

- FFX gaps are heterogeneous across sizes. Small/medium/large sampled rows: `+0.005`
  to `+0.016`. Large/huge sampled test: `+0.009` to `+0.031` — the huge sampled test gap
  (`+0.031`) is notably larger than the rest. Mixed rows: `+0.013` to `+0.024` for
  small/medium/large; huge mixed train is **−0.0054** (current beats INLA on FFX).
- σ remains consistently worse than INLA across all sizes and partitions, especially on
  sampled rows. Recent diagnostics show that plugging in true σ or INLA σ does not close
  the FFX gap, so σ is not the primary next optimization target.
- BLUP is mixed: current beats INLA slightly on huge sampled valid (`0.6195 vs 0.6223`)
  and several smaller sampled rows, but lags on mixed rows and huge sampled test.
- CPU runtime is acceptable for this phase. The next patches should prioritize FFX
  movement; speed work is explicitly deferred until the accuracy ceiling is clearer.

Full 8192-Row Current Benchmark
-------------------------------

Full available Poisson rows per cell (`8192` indices), mixed train over two training
epochs and sampled valid/test. These are current-method results only; the INLA reference
table above remains the first-1000 comparison because the R-INLA fits were produced on
that subset.

| Dataset | part | N | FFX | σ | BLUP | ms/ds | EB accept | σ cap | grid accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 8192 | 0.1668 | 0.3592 | 0.5104 | 74.3 | 0.883 | 0.000 | 0.974 |
| small-p-sampled | valid | 8192 | 0.2024 | 0.4225 | 0.5149 | 71.0 | 0.892 | 0.000 | 0.984 |
| small-p-sampled | test | 8192 | 0.1970 | 0.4195 | 0.5169 | 72.7 | 0.878 | 0.000 | 0.985 |
| medium-p-mixed | train | 8192 | 0.1706 | 0.4184 | 0.4967 | 128.2 | 0.781 | 0.086 | 0.997 |
| medium-p-sampled | valid | 8192 | 0.2078 | 0.4436 | 0.5516 | 127.2 | 0.819 | 0.077 | 0.998 |
| medium-p-sampled | test | 8192 | 0.2062 | 0.4517 | 0.5631 | 132.9 | 0.799 | 0.078 | 0.998 |
| large-p-mixed | train | 8192 | 0.2076 | 0.5085 | 0.7335 | 152.5 | 0.757 | 0.116 | 0.996 |
| large-p-sampled | valid | 8192 | 0.2224 | 0.5151 | 0.6077 | 148.0 | 0.755 | 0.122 | 0.997 |
| large-p-sampled | test | 8192 | 0.2254 | 0.5228 | 0.6482 | 160.0 | 0.770 | 0.113 | 0.997 |
| huge-p-mixed | train | 8192 | 0.2109 | 0.5435 | 0.5904 | 204.6 | 0.707 | 0.157 | 0.996 |
| huge-p-sampled | valid | 8192 | 0.2387 | 0.5910 | 0.6774 | 226.8 | 0.743 | 0.159 | 0.995 |
| huge-p-sampled | test | 8192 | 0.2650 | 0.5890 | 0.7061 | 208.0 | 0.751 | 0.164 | 0.996 |

Notes:

- The 8192-row aggregate numbers do not show a single cell-level collapse. FFX remains
  strongest on small/medium and weakest on huge sampled test; σ and BLUP degrade with
  size, especially large/huge sampled and large mixed BLUP.
- Index-level outlier logging was not captured during this aggregate run. Do not rerun a
  separate full 12-cell row diagnostic just to recover indices; instead add row-level
  logging to the benchmark path when the next full pass is needed, or run targeted
  row-index diagnostics only on suspicious cells such as `huge-p-sampled:test` and
  `large-p-mixed:train`.
- Stage-level profiling can start after either targeted suspicious-cell row checks or an
  integrated benchmark+row-log pass confirms there is no small set of dominating outlier
  indices.

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

Accuracy prototype results:

- Multi-state VG covariance-temperature averaging is safe but tiny. On first-1000
  small/medium rows it moved FFX only by about `-0.0006` to `+0.0001`:
  `small-mixed 0.2076 -> 0.2075`, `small-valid 0.2326 -> 0.2324`,
  `small-test 0.2107 -> 0.2108`, `medium-mixed 0.1807 -> 0.1801`,
  `medium-valid 0.2304 -> 0.2304`, `medium-test 0.2383 -> 0.2382`. Keep it
  opt-in; it is not enough to justify default complexity.
- Current-path row diagnostics over 2k representative rows found weak direct
  correlation between FFX RMSE and `d`/`q` (`+0.005`/`+0.046`) but stronger
  correlation with BLUP and σ RMSE (`+0.382`/`+0.276`). Many worst FFX rows had
  zero adaptive VG acceptances and high σ-average effective candidate counts, so the
  remaining failures are not a simple high-dimensional gate or scalar-σ uncertainty
  problem.
- VG line-search polish over dampings `(0.25, 0.5, 0.75, 1.0)` is also FFX-neutral:
  first-1000 small/medium rows moved from `0.2076 -> 0.2075`,
  `0.2326 -> 0.2323`, `0.2107 -> 0.2110`, `0.1807 -> 0.1807`,
  `0.2304 -> 0.2304`, `0.2383 -> 0.2382`. It improved medium σ/BLUP
  (`medium-valid σ 0.4500 -> 0.4460`, BLUP `0.5497 -> 0.5490`; `medium-test σ
  0.4388 -> 0.4343`, BLUP `0.5504 -> 0.5499`) but does not materially close the
  INLA FFX gap.

Next Directions
---------------

Primary patch candidates:

0. **Speed profiling and ablation, once outliers are cleared.**
   The 8192-row current benchmark is accuracy-stable at the cell level but currently runs
   around `70-227 ms/ds` on CPU. The next production-relevant step is stage-level timing
   and accuracy-preserving ablations, but only after targeted row-index checks rule out a
   small number of problematic rows driving the remaining large/huge errors.

1. **Hard-row targeted high-budget VG / convergence probe.**
   Extra VG budget was the only clear hard-row lever, while cheap state averaging and
   line-search polish were nearly flat. The next informative test is to apply high-budget
   VG only to rows flagged by poor BLUP/σ fit, large FFX row RMSE proxy, or no adaptive
   acceptances, then check whether the global FFX gap can move without making every row
   expensive.

2. **Better q(u) posterior approximation.**
   The remaining FFX error tracks BLUP/σ quality more than dimension or scalar σ
   uncertainty. A serious next architecture is a richer latent Gaussian update, for
   example low-rank/full per-group `V_g` handling already exists but may need better
   calibration, or a local variational/Laplace update that targets posterior means rather
   than conditional modes.

3. **Revisit objective calibration against INLA means.**
   Target improvements do not reliably move FFX, and one-shot β posterior-mean corrections
   failed. If high-budget targeted VG still leaves the FFX gap, test whether the VG/ELBO
   target is systematically miscalibrated relative to INLA posterior means by comparing
   hard-row target ranks, posterior variance correction, and INLA β deltas where fitted
   rows are available.

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
