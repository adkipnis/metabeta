Poisson GLMM Plan
=================

Last updated: 2026-05-21 (sequential Poisson patch tests)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA is a
reference only; R-INLA-as-backend and full PyTorch INLA remain out of scope.

Current Default
---------------

The retained Poisson path is:

1. RAW/PQL Poisson GLMM initialization.
2. Diagonal single-mode Laplace EB over β and diagonal σ_rfx.
3. Fixed-budget diagonal-Σ joint Laplace-PIRLS over β/u/σ.
4. Marginal-mean β correction with `min_d=1`.
5. Conservative full-candidate PIRLS σ grid with scales `(0.5, 0.75, 1.0)`.
6. Scalar Laplace-weighted σ averaging with local fixed-σ β/u PIRLS refresh.
7. Variational-Gaussian posterior-mean refinement with compressed budget
   `outer=5, inner=3, final=2`.
8. VG-centered scalar σ averaging with β-only weighted output.

The final VG-centered averaging pass approximates local hyperparameter integration over
diagonal σ scales and writes back weighted β only. This is now the default because the
compressed VG schedule materially improved medium/large mixed rows and modestly improved
sampled rows without a broad regression.

Important method names:

- `current` / `default`: retained path with compressed VG plus VG-centered scalar σ
  averaging.
- `poisson_laplace_pirls_sigma_avg`: explicit scalar σ averaging path.
- `poisson_laplace_pirls_sigma_avg_is`: intercept-vs-slope σ averaging prototype.
- `poisson_laplace_target_refine`: older fixed-σ β/u autograd ablation under the diagonal
  Laplace target; kept for diagnostics only.
- `poisson_variational_gaussian`: explicit iterated Gaussian variational posterior-mean
  solver with conservative σ blending.
- `poisson_variational_gaussian_sigma_avg`: `poisson_variational_gaussian` plus
  ELBO-weighted scalar σ averaging around the VG state; writes back weighted β only by
  default. This now matches the default Poisson path.
- `poisson_variational_gaussian_sigma_avg_is`: intercept-vs-slope version of the same
  VG-centered averaging prototype; slightly stronger but slower.
- `poisson_laplace_pirls_full` / `poisson_laplace_pirls_full_beta`: full-Σ prototypes;
  stable but worse than the diagonal default in the tested rows.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU rerun on 2026-05-21 for the selected mixed/valid
rows. Lower NRMSE is better. INLA values are the current first-1000 diagonal R-INLA
references.

| Dataset | part | default FFX | INLA FFX | FFX gap | default σ | INLA σ | default BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2076 | 0.1835 | +0.0241 | 0.4272 | 0.3404 | 0.5205 | 0.4936 | 69.2 |
| small-p-sampled | valid | 0.2327 | 0.2276 | +0.0051 | 0.4784 | 0.4356 | 0.5325 | 0.5309 | 83.8 |
| medium-p-mixed | train | 0.1815 | 0.1675 | +0.0140 | 0.4003 | 0.3214 | 0.5072 | 0.4789 | 121.1 |
| medium-p-sampled | valid | 0.2363 | 0.2146 | +0.0217 | 0.4827 | 0.4209 | 0.5576 | 0.5618 | 124.7 |
| large-p-mixed | train | 0.1955 | 0.1778 | +0.0177 | 0.4365 | 0.3076 | 0.5425 | 0.5001 | 138.4 |
| large-p-sampled | valid | 0.2924 | 0.2467 | +0.0457 | 0.4990 | 0.4232 | 0.6204 | 0.5870 | 134.1 |

Interpretation:

- The compressed VG default makes mixed medium/large rows much closer to INLA for FFX.
- Sampled large rows remain the clearest FFX gap (`~0.046` NRMSE on valid), while σ is
  still consistently worse than INLA.
- Runtime is now above the original ideal on CPU for medium/large rows, so further
  patches need a clear FFX gain or a cheaper approximation to the same VG behavior.

What We Retired
---------------

- Direct VG from PQL: fast but much worse than using the older EB/PIRLS/grid path as the
  initializer. On sampled-valid rows, FFX regressed to `0.320/0.351/0.415` for
  small/medium/large after σ averaging.
- VG-target autograd polish: tiny small-row FFX gain but medium/large regressions
  (`0.2453 -> 0.2423` medium valid was not enough, large valid `0.2975 -> 0.2977`; with
  compressed VG, medium valid regressed to `0.2625`). Removed from the production path.
- β-only σ grid: replaced by full-candidate grid and scalar weighted σ averaging.
- β-only AGQ optimizer: slower and did not close the main gap.
- More gate tuning around the old EB/grid path: low expected value.
- Broad σ inflation grids: rare but large σ outliers.
- Full-Σ PIRLS as default: stable and similarly fast, but worse than the diagonal default.
- More scalar covariance-update tuning: only small gains and did not improve σ/BLUP enough.

Active Prototypes
-----------------

### Iterated Gaussian Variational Posterior-Mean Solver

`poisson_variational_gaussian` starts from the default path, then runs an iterated
variational-EM style solver for `q(u_g)=N(m_g,V_g)`. Each cycle updates β/m against the
expected Poisson mean `exp(Xβ + Zm_g + 0.5 * diag(Z V_g Z'))`, refreshes `V_g` at the
updated state, updates diagonal σ from `m_g² + diag(V_g)`, and refreshes `V_g` again after
the σ move. Diagonal σ is blended conservatively with the previous σ estimate
(`sigma_blend=0.25`). The current fixed budget is `outer=5, inner=3, final=2`.

First 1000 rows, sequential CPU run after the coherent V/σ refresh patch.

| Dataset | part | default FFX | var-Gauss FFX | default σ | var-Gauss σ | default BLUP | var-Gauss BLUP | var-Gauss ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2151 | 0.2079 | 0.4287 | 0.4273 | 0.5222 | 0.5205 | 38.0 |
| small-p-sampled | valid | 0.2378 | 0.2335 | 0.4756 | 0.4785 | 0.5355 | 0.5325 | 35.3 |
| small-p-sampled | test | 0.2119 | 0.2111 | 0.4421 | 0.4393 | 0.5232 | 0.5208 | 35.9 |
| medium-p-mixed | train | 0.2438 | 0.2334 | 0.4198 | 0.4008 | 0.5129 | 0.5073 | 69.0 |
| medium-p-sampled | valid | 0.3057 | 0.2450 | 0.5654 | 0.4930 | 0.6115 | 0.5723 | 72.6 |
| medium-p-sampled | test | 0.2831 | 0.2740 | 0.4723 | 0.4448 | 0.5560 | 0.5510 | 68.8 |
| large-p-mixed | train | 0.3436 | 0.2432 | 0.5536 | 0.4549 | 0.6472 | 0.5512 | 79.6 |
| large-p-sampled | valid | 0.4131 | 0.3026 | 0.5261 | 0.4996 | 0.6318 | 0.6211 | 79.5 |
| large-p-sampled | test | 0.4358 | 0.2889 | 0.5124 | 0.4714 | 0.7651 | 0.7213 | 90.5 |

Interpretation:

- Stronger fixed-budget iteration is the first large FFX improvement since scalar σ
  averaging. The largest gains are on the high-dimensional rows that were farthest from
  INLA.
- The prototype improves FFX on every tested small/medium/large row and usually improves
  σ/BLUP too. σ is no longer the main decision criterion.
- Runtime stays inside the target envelope for these first-1000 CPU rows. The path should
  be validated on huge rows before becoming the retained default.
- Remaining gap to INLA is still meaningful on large sampled rows, so the next improvement
  should extend posterior/hyperparameter averaging around the variational state rather
  than tune old gates.

### VG-Centered σ/Hyperparameter Averaging

`poisson_variational_gaussian_sigma_avg` starts from the iterated VG state, evaluates a
small log-σ grid, runs local VG β/m/V refreshes under each candidate σ, scores candidates
by the variational target, and returns the weighted β mean. The default output mode is now
β-only because it preserves the FFX gains while avoiding σ/BLUP churn from averaged σ
writeback.

First 1000 rows, sequential CPU run. The VG-centered σ average uses scalar scales
`(0.5, 0.75, 1.0, 1.3333, 2.0)` and `temperature=2.0`.

| Dataset | part | var-Gauss FFX | VG σ-avg FFX | var-Gauss σ | VG σ-avg σ | var-Gauss BLUP | VG σ-avg BLUP | VG σ-avg ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2079 | 0.2076 | 0.4273 | 0.4273 | 0.5205 | 0.5205 | 52.3 |
| small-p-sampled | valid | 0.2335 | 0.2327 | 0.4785 | 0.4785 | 0.5325 | 0.5325 | 51.3 |
| small-p-sampled | test | 0.2111 | 0.2108 | 0.4393 | 0.4393 | 0.5208 | 0.5208 | 46.6 |
| medium-p-mixed | train | 0.2334 | 0.2308 | 0.4008 | 0.4008 | 0.5073 | 0.5073 | 84.1 |
| medium-p-sampled | valid | 0.2450 | 0.2453 | 0.4930 | 0.4930 | 0.5723 | 0.5723 | 86.3 |
| medium-p-sampled | test | 0.2740 | 0.2738 | 0.4448 | 0.4448 | 0.5510 | 0.5510 | 83.6 |
| large-p-mixed | train | 0.2432 | 0.2374 | 0.4549 | 0.4549 | 0.5512 | 0.5512 | 99.4 |
| large-p-sampled | valid | 0.3026 | 0.2975 | 0.4996 | 0.4996 | 0.6211 | 0.6211 | 99.5 |
| large-p-sampled | test | 0.2889 | 0.2796 | 0.4714 | 0.4714 | 0.7213 | 0.7213 | 107.2 |

Interpretation:

- The scalar VG-centered average gives consistent but modest additional FFX gains on large
  rows, small gains on small rows, and no useful gain on `medium-p-sampled valid`.
- β-only output is the clean setting: `beta_sigma` output produced a severe σ regression
  on `medium-p-sampled valid` (`0.4930 -> 0.6561`) without FFX benefit.
- Intercept-vs-slope averaging is slightly better on representative large rows but slower:
  `large-p-mixed` FFX `0.2374 -> 0.2371`, `large-p-sampled test` `0.2796 -> 0.2788`.
  Keep it as an ablation rather than a default.
- This is a useful incremental patch, not a full answer to the INLA gap. Large sampled FFX
  remains around `0.28-0.30`, still materially worse than INLA `~0.22-0.25`.

### Intercept-vs-Slope σ Averaging

This usually ties or slightly beats scalar σ averaging for FFX and often improves σ/BLUP,
but gains are small and not uniform. Keep it as an ablation until diagnostics show a clear
case where separate intercept/slope hyperparameter uncertainty matters.

### Direct Laplace-Target β/u Refinement

`poisson_laplace_target_refine` holds σ fixed and takes small autograd Adam steps on β/u
against the diagonal Laplace target, accepting per row only if the target improves.

Older quick 1k validation:

| Dataset | part | default FFX | target refine FFX | default σ | target refine σ | default BLUP | target refine BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2151 | 0.2131 | 0.4287 | 0.4287 | 0.5222 | 0.5224 |
| small-p-sampled | valid | 0.2378 | 0.2365 | 0.4756 | 0.4756 | 0.5355 | 0.5344 |
| small-p-sampled | test | 0.2119 | 0.2106 | 0.4421 | 0.4421 | 0.5232 | 0.5222 |
| medium-p-mixed | train | 0.2438 | 0.2427 | 0.4198 | 0.4198 | 0.5129 | 0.5128 |

The signal was positive but too small to close the INLA gap. A VG-target polish tested
after the current VG σ-average did not improve the medium/large rows, so posthoc fixed-σ
β/u polishing remains low priority.

Targeted Diagnostic
-------------------

Focused script:

```bash
uv run python -u experiments/analytical/glmm_poisson_variational_diagnostic.py \
    --combos medium-p-sampled:valid large-p-sampled:test \
    --max-datasets 1000 --batch-size 32
```

Representative aggregate over the two hard sampled rows:

| variant | FFX | σ | BLUP | ms/ds |
| --- | ---: | ---: | ---: | ---: |
| current | 0.3830 | 0.5390 | 0.7025 | 70.0 |
| var-Gauss, old σ blend 0.5 | 0.3580 | 0.5316 | 0.6707 | 73.8 |
| offline β-only var-Gauss | 0.3580 | 0.5390 | 0.7025 | 73.8 |
| offline β+σ var-Gauss | 0.3580 | 0.5316 | 0.7025 | 73.8 |
| var-Gauss, σ blend 0.25 | 0.3578 | 0.5248 | 0.6719 | 72.9 |
| var-Gauss, stronger σ prior | 0.3580 | 0.5262 | 0.6714 | 72.6 |
| var-Gauss without scalar σ averaging | 0.3633 | 0.6264 | 0.6586 | 66.9 |

Findings:

- β-only writeback captures the FFX gain, confirming that posterior-mean β geometry is
  the main useful component of the variational update.
- Full writeback is still preferable in aggregate because it also improves σ/BLUP versus
  current.
- Lower σ blending (`0.25`) is the best tested compromise: same or slightly better FFX,
  better σ than the old blend, and still better BLUP than current.
- Running the variational update without the scalar σ-averaging stage is worse for FFX/σ,
  so the current default remains a useful initializer.
- Row-level FFX gains correlate with larger existing marginal variance correction
  (`corr(delta_ffx, current_sigma_q95)=-0.54`) and with ELBO target improvement
  (`corr(delta_ffx, target_delta)=-0.42`).

Implication:

- Continue the variational/posterior-mean architecture.
- β-only writeback is the preferred setting for post-VG σ averaging; it captures the FFX
  gain without causing σ/BLUP regressions.
- The next likely useful guard is targeted σ writeback suppression for extreme marginal
  variance rows, not more generic gate tuning.

### Hard-Row Oracle/VG Diagnostic

Focused script:

```bash
uv run python -u experiments/analytical/glmm_poisson_hard_row_diagnostic.py \
    --combos medium-p-sampled:valid large-p-sampled:test \
    --max-datasets 1000 --top-k 64 --batch-size 32 --rank-by vg_rmse
```

This ranks the first 1000 rows of each hard sampled cell by row-level FFX RMSE under
`poisson_variational_gaussian_sigma_avg`, then reruns the same selected rows with true-σ,
INLA-σ, wider σ averaging, and high-budget VG variants. Aggregate over the 64 selected
hard rows:

| variant | FFX | σ | BLUP | ms/ds | N_eff | best scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 1.2441 | 1.4276 | 1.7256 | 211.2 | n/a | n/a |
| VG σ-avg | 0.6285 | 0.9487 | 0.9049 | 261.8 | 4.50 | 0.99 |
| wider VG σ-avg | 0.6302 | 0.9487 | 0.9049 | 291.2 | 5.90 | 0.98 |
| high-budget VG | 0.5377 | 0.5517 | 0.6489 | 311.1 | n/a | n/a |
| high-budget VG + wider σ-avg | 0.5352 | 0.5517 | 0.6489 | 355.1 | 6.04 | 0.94 |
| true-σ fixed refresh | 0.6380 | 0.0000 | 0.9049 | 261.8 | 1.49 | 1.00 |
| INLA-σ fixed refresh | 0.6839 | 1.3958 | 0.9049 | 261.8 | 1.18 | 1.00 |
| INLA | 1.2598 | 1.3958 | 1.1538 | n/a | n/a | n/a |

Additional check with `--rank-by gap_to_inla` produced the same qualitative result:
wider σ averaging and σ plug-ins were not the useful lever; high-budget VG refinement was.

Findings:

- The remaining β failures are not mostly caused by the diagonal σ point estimate.
  Replacing σ by true σ and locally refreshing β does not improve FFX on the selected
  hard rows (`0.6285 -> 0.6380`).
- INLA σ plug-in is worse for our local VG refresh (`0.6839` FFX), so INLA's advantage is
  not transferable as a standalone σ estimate.
- Wider VG σ averaging increases effective candidate count but does not improve FFX
  (`0.6285 -> 0.6302`). The current scalar grid is not the bottleneck.
- More fixed-budget VG iteration is the only tested lever with a material hard-row gain:
  `0.6285 -> 0.5377`, while also improving σ and BLUP. The extra wider σ averaging on top
  is negligible (`0.5377 -> 0.5352`).
- Only about 5% of the top VG-error rows have row-level VG error above INLA error. The
  aggregate INLA gap is therefore probably broad distributional calibration, not a small
  set of obvious β outliers.

Implication:

- Stop prioritizing wider σ grids, INLA/true-σ plug-ins, and scalar hyperparameter-grid
  tuning for FFX. They are useful diagnostics but not the main path.
- The next serious Poisson patch should improve the posterior-mean/VG update itself:
  better β/m/V convergence, target calibration, or a more coherent variational-EM loop.
- High-budget VG is a useful oracle for direction, but it is too slow to promote as-is.
  Use it to identify which additional iterations matter, then compress that behavior into
  a smaller update.

Most Promising Architectural Change
-----------------------------------

The most plausible route to FFX around `~0.20` remains a posterior-mean-oriented
Poisson variational/Laplace update, not another posthoc correction around the current
conditional mode. The iterated `poisson_variational_gaussian` prototype validates this:
large-row FFX drops from `0.34-0.44` to `0.24-0.30`, much closer to INLA but not yet
matching it.

Prototype target:

```text
q(u_g) = N(m_g, V_g)

E_q[y_i mean] =
    exp(x_i β + z_i m_g + 0.5 * z_i V_g z_i)

repeat fixed budget:
    update β against expected Poisson mean
    update m_g, V_g by local Laplace/Newton steps
    update Σ = average_g(m_g m_g' + V_g) with prior shrinkage
```

Why this is the next serious candidate:

- INLA likely wins through posterior-mean geometry and target calibration, while our
  current path is still a shallow approximation to that geometry.
- The Poisson log link makes β highly sensitive to uncertainty in random effects through
  `exp(0.5 * z'Σz)`.
- Scalar σ averaging helps modestly, but the hard-row oracle diagnostic shows that simple
  σ-scale integration is no longer the bottleneck.
- Full-Σ covariance and true/INLA σ plug-ins did not help, so the missing piece is more
  likely the β/m/V posterior-mean update and objective calibration than covariance shape.
- Direct fixed-σ Laplace target refinement helps only marginally, so optimizing the same
  conditional mode harder is unlikely to reach INLA territory.

Next refinements for this path:

- Diagnose high-budget VG iteration trajectories on hard rows: which outer cycles reduce
  β error, whether β or m updates dominate, and whether σ movement is necessary.
- Prototype a compressed VG update that reuses the effective high-budget sequence without
  running all 10 outer / 4 inner steps, e.g. two stronger β/m refreshes plus one coherent
  V/σ refresh and acceptance by the variational target.
- Test objective calibration variants for the VG target: prior strength, variance
  correction clipping, and damping schedules. Keep the test row set fixed to the hard-row
  diagnostic above.
- Keep σ-grid expansion and σ-writeback guards low priority unless future variants start
  trading FFX gains against severe σ/BLUP regressions.

Secondary direction:

- Move direct Laplace-target β/u refinement inside each σ candidate before weighting. This
  is smaller than the variational path and may improve scalar averaging, but current
  fixed-σ gains suggest it is unlikely to close the large-row gap by itself.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_variational_gaussian poisson_variational_gaussian_sigma_avg \
    --sizes small medium large \
    --batch-size 32 --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
