Poisson GLMM Plan
=================

Last updated: 2026-05-20 (PIRLS full-grid is the retained Poisson path)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA is a
reference only; R-INLA-as-backend and full PyTorch INLA remain out of scope.

Retained Model
--------------

The default Poisson analytical path is now:

1. RAW/PQL Poisson GLMM initialization.
2. Diagonal single-mode Laplace EB over β and diagonal σ_rfx.
3. Fixed-budget diagonal-Σ joint Laplace-PIRLS over β/u/σ.
4. Marginal-mean β correction with `min_d=1`.
5. Conservative full-candidate PIRLS σ grid with scales `(0.5, 0.75, 1.0)`.

The final grid scores and writes back the full candidate: β, σ, BLUPs, BLUP variances, and
`Psi_lap`. It deliberately allows only shrinkage or fixed-σ β/u re-synchronization. Broad
inflation scales and the older β-only σ grid were retired after producing worse σ
tradeoffs without meaningful FFX gain.

Benchmark method names:

- `current` / `default`: retained full-grid path.
- `poisson_laplace_pirls_full_grid`: explicit retained path.
- `poisson_laplace_pirls_beta`: ablation without the final full-candidate grid.
- `poisson_laplace_pirls_diag`: ablation without marginal β and final grid.
- `poisson_marginal_beta` and `poisson_eb`: historical ablations only.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU runs on 2026-05-20. Lower NRMSE is better.
INLA values are the current first-1000 diagonal R-INLA references.

| Dataset | part | full-grid FFX | INLA FFX | full-grid σ | INLA σ | full-grid BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2214 | 0.1835 | 0.4529 | 0.3404 | 0.5245 | 0.4936 | 37.9 |
| small-p-sampled | valid | 0.2523 | 0.2276 | 0.4919 | 0.4356 | 0.5384 | 0.5309 | 34.2 |
| small-p-sampled | test | 0.2205 | 0.1997 | 0.4618 | 0.3966 | 0.5270 | 0.5281 | 36.4 |
| medium-p-mixed | train | 0.2670 | 0.1675 | 0.4360 | 0.3214 | 0.5199 | 0.4789 | 66.7 |
| medium-p-sampled | valid | 0.3385 | 0.2146 | 0.5229 | 0.4209 | 0.6334 | 0.5618 | 68.8 |
| medium-p-sampled | test | 0.2968 | 0.2267 | 0.5016 | 0.3883 | 0.5603 | 0.5849 | 64.6 |
| large-p-mixed | train | 0.3821 | 0.1778 | 0.6401 | 0.3076 | 0.6980 | 0.5001 | 73.0 |
| large-p-sampled | valid | 0.4422 | 0.2467 | 0.5475 | 0.4232 | 0.6414 | 0.5870 | 69.1 |
| large-p-sampled | test | 0.4608 | 0.2186 | 0.6215 | 0.3439 | 0.8082 | 0.5618 | 78.3 |

Validation summary:

- Full-grid beats the previous Poisson path and PIRLS+β on FFX and BLUP across small,
  medium, and large rows.
- On large rows it also improves σ and runtime versus both adjacent baselines.
- The residual FFX gap to INLA is about `0.020-0.038` on small rows, `0.070-0.124` on
  medium rows, and `0.196-0.242` on large rows.
- σ remains materially worse than INLA, especially on large rows. This is now the most
  plausible source of the remaining FFX gap through the Poisson log-link marginal mean.

Covariance-Update Sanity Sweep
------------------------------

First 1000 mixed/train rows on 2026-05-20, using the retained full-grid path and changing
one PIRLS covariance/update knob at a time:

| setting | small FFX | small σ | small BLUP | medium FFX | medium σ | medium BLUP | ms/ds range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.2214 | 0.4529 | 0.5245 | 0.2670 | 0.4360 | 0.5199 | 35.7-58.2 |
| `sigma_prior_weight=8` | 0.2182 | 0.4651 | 0.5263 | 0.2700 | 0.4454 | 0.5206 | 39.0-64.4 |
| `sigma_blend=0.25` | 0.2178 | 0.4586 | 0.5248 | 0.2891 | 0.4612 | 0.5198 | 35.0-58.7 |
| `damping=0.35` | 0.2344 | 0.4460 | 0.5275 | 0.3060 | 0.4498 | 0.5247 | 37.8-65.7 |
| `final=4` | 0.2166 | 0.4608 | 0.5257 | 0.2631 | 0.4341 | 0.5212 | 37.6-61.2 |
| `final=6` | 0.2160 | 0.4607 | 0.5261 | 0.2602 | 0.4468 | 0.5216 | 38.1-65.8 |

Interpretation:

- Stronger covariance shrinkage or slower covariance blending does not close the medium
  FFX gap and usually worsens σ.
- Lower damping hurts FFX, so the main issue is not obvious PIRLS overshoot.
- More final fixed-σ PIRLS steps give a small FFX gain, but the gain is much smaller than
  the INLA gap and trades against σ/BLUP/runtime. Keep this as a low-risk knob, not a main
  research path.
- Further scalar tuning of the diagonal covariance update is low expected value.

Grid Diagnostic
---------------

Across 6000 small/medium diagnostic rows, conservative full-grid acceptance was
`0.982-0.999` by cell. About 82% of accepted rows selected scale `1.0`, so the grid mostly
acts as a fixed-σ β/u re-sync after marginal β correction. Shrinkage scales were less
common but useful:

| scale | rows | base FFX | grid FFX | base σ | grid σ | base BLUP | grid BLUP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 179 | 0.4058 | 0.2918 | 0.8973 | 0.7484 | 0.7020 | 0.6804 |
| 0.75 | 856 | 0.3024 | 0.2218 | 0.6413 | 0.5946 | 0.6741 | 0.6398 |
| 1.0 | 4922 | 0.3321 | 0.2736 | 0.4642 | 0.4642 | 0.5714 | 0.5431 |

Rejected
--------

- Old β-only σ grid: retired from the implementation. It scored σ candidates but wrote
  back only β, and was superseded by full-candidate writeback.
- β-only AGQ optimizer: retired from the implementation. It was slower and did not close
  the main gap.
- More gate tuning around the old EB/grid path: low expected value.
- Broad σ inflation grids: rejected after rare but large σ outliers.
- Full PyTorch INLA, EP, broad multi-start optimization: too much complexity for this
  analytical path.

Next Directions
---------------

1. **Do not spend more effort on scalar diagonal covariance tuning.**

   Current update:

   ```text
   sigma_j^2 <- (sum_g (u_gj^2 + diag(A_g^-1)_j) + nu0 * sigma0_j^2) / (m + nu0)
   ```

   The sanity sweep found only small FFX gains from extra final fixed-σ PIRLS steps and no
   convincing σ/BLUP improvement. Leave defaults unchanged for now; `final=4` or `final=6`
   can be retested later if a larger architecture change makes convergence the bottleneck.

2. **Move the next serious patch to full-Σ joint PIRLS.**

   The current diagonal-Σ solver is still substantially behind INLA on FFX and σ,
   especially from medium upward. If covariance tuning is weak, the likely missing piece is
   latent covariance geometry, not another scalar correction. Full `Σ` is cheap for
   `q <= 5`, but it should be estimated inside the joint β/u/Σ PIRLS loop, not as a
   posthoc marginal β correction.

3. **Keep one accept/reject target.**

   Use the diagonal Laplace target for joint candidates:

   ```text
   J = poisson_nll(y | beta, u)
       + 0.5 * sum_g u_g' Sigma^-1 u_g
       + 0.5 * m * log|Sigma|
       + 0.5 * sum_g log|Z_g' W_g Z_g + Sigma^-1|
       + priors
   ```

   The same target should score current EB/full-grid candidates and future full-Σ PIRLS
   candidates, so we can compare architectures rather than gate heuristics.

4. **If full-Σ PIRLS still leaves a large FFX gap, ask for/try architectural alternatives.**

   The main alternatives to discuss or prototype are cheap hyperparameter averaging around
   the PIRLS mode, Laplace-target β optimization with curvature terms, or a limited
   multi-candidate solver. Avoid broad posthoc gates unless they are selected by the same
   coherent objective.

5. **Run huge-row validation before 8k.**

   Full 8k Poisson benchmarks remain postponed until one more meaningful accuracy patch or
   complete huge-row validation lands.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods raw poisson_eb poisson_marginal_beta \
    poisson_laplace_pirls_diag poisson_laplace_pirls_beta current \
    --sizes small medium large --batch-size 32 --max-datasets 1000

uv run python -u experiments/analytical/glmm_poisson_pirls_grid_diagnostic.py \
    --sizes small medium --max-datasets 1000 --batch-size 32

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
