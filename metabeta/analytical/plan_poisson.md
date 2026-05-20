Poisson GLMM Plan
=================

Last updated: 2026-05-21 (scalar σ averaging promoted; Laplace-target refinement prototype)

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
6. Scalar Laplace-weighted σ averaging with local fixed-σ β/u PIRLS refresh.

The conservative grid scores and writes back the full candidate: β, σ, BLUPs, BLUP
variances, and `Psi_lap`. The scalar averaging pass then approximates local
hyperparameter integration over diagonal σ scales and writes back weighted β/σ plus BLUPs
from the best-scoring candidate. Intercept-vs-slope σ averaging remains an opt-in
prototype because its gains are small and not uniformly better than scalar averaging.

Benchmark method names:

- `current` / `default`: retained full-grid path plus scalar σ averaging.
- `poisson_laplace_pirls_full_grid`: explicit retained path.
- `poisson_laplace_pirls_full`: full-Σ joint PIRLS prototype without marginal β.
- `poisson_laplace_pirls_full_beta`: full-Σ joint PIRLS plus full-Ψ marginal β.
- `poisson_laplace_pirls_beta`: ablation without the final full-candidate grid.
- `poisson_laplace_pirls_diag`: ablation without marginal β and final grid.
- `poisson_laplace_pirls_sigma_avg`: explicit scalar Laplace-weighted σ averaging path.
- `poisson_laplace_pirls_sigma_avg_is`: retained full-grid path plus intercept-vs-slope
  Laplace-weighted σ averaging prototype.
- `poisson_laplace_target_refine`: current default plus 1-2 direct β/u autograd steps
  under the diagonal Laplace target.
- `poisson_marginal_beta` and `poisson_eb`: historical ablations only.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU runs on 2026-05-20. Lower NRMSE is better.
INLA values are the current first-1000 diagonal R-INLA references.

| Dataset | part | default FFX | INLA FFX | default σ | INLA σ | default BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2151 | 0.1835 | 0.4287 | 0.3404 | 0.5222 | 0.4936 | 33.5 |
| small-p-sampled | valid | 0.2378 | 0.2276 | 0.4756 | 0.4356 | 0.5355 | 0.5309 | 36.7 |
| small-p-sampled | test | 0.2119 | 0.1997 | 0.4421 | 0.3966 | 0.5232 | 0.5281 | 33.8 |
| medium-p-mixed | train | 0.2438 | 0.1675 | 0.4198 | 0.3214 | 0.5129 | 0.4789 | 64.1 |
| medium-p-sampled | valid | 0.3057 | 0.2146 | 0.5654 | 0.4209 | 0.6115 | 0.5618 | 77.6 |
| medium-p-sampled | test | 0.2831 | 0.2267 | 0.4723 | 0.3883 | 0.5560 | 0.5849 | 71.6 |
| large-p-mixed | train | 0.3436 | 0.1778 | 0.5536 | 0.3076 | 0.6472 | 0.5001 | 82.4 |
| large-p-sampled | valid | 0.4131 | 0.2467 | 0.5261 | 0.4232 | 0.6318 | 0.5870 | 85.2 |
| large-p-sampled | test | 0.4358 | 0.2186 | 0.5124 | 0.3439 | 0.7651 | 0.5618 | 86.7 |

Validation summary:

- Scalar σ averaging beats the retained full-grid path on FFX for every sampled and large
  row tested, and usually improves σ/BLUP as well.
- The residual FFX gap to INLA is about `0.010-0.032` on small rows, `0.056-0.091` on
  medium rows, and `0.166-0.217` on large rows.
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

Full-Σ PIRLS Prototype
----------------------

Implemented as a separate non-default path on 2026-05-20:

```text
poisson_laplace_pirls_full
    EB initializer -> full-Σ joint β/u/Σ PIRLS

poisson_laplace_pirls_full_beta
    EB initializer -> full-Σ joint β/u/Σ PIRLS -> full-Ψ marginal β correction
```

The full-Σ covariance update uses posterior second moments:

```text
Σ <- (sum_g (u_g u_g' + A_g^-1) + nu0 * Σ0) / (m + nu0)
```

with eigenvalue clipping and a diagonal prior anchor. The candidate is accepted by the same
full Laplace target shape used for the diagonal solver.

First 1000 mixed/train rows:

| method | small FFX | small σ | small BLUP | medium FFX | medium σ | medium BLUP | ms/ds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `poisson_laplace_pirls_full` | 0.3039 | 0.4677 | 0.5263 | not run | not run | not run | 34.6 |
| `poisson_laplace_pirls_full_beta` | 0.2582 | 0.4677 | 0.5263 | 0.3284 | 0.5060 | 0.5178 | 33.2-56.4 |
| `current` | 0.2214 | 0.4529 | 0.5245 | 0.2670 | 0.4360 | 0.5199 | 34.2-57.8 |

Interpretation:

- Full-Σ PIRLS is numerically stable and similar in speed to the diagonal path.
- Full-Ψ marginal β helps the full-Σ candidate, but not enough to beat the retained
  diagonal full-grid path.
- The prototype does not support making full-Σ the default. The likely issue is not merely
  missing off-diagonal covariance; the posterior/hyperparameter averaging gap to INLA is
  still present.

Laplace-Weighted σ Averaging Prototypes
---------------------------------------

Implemented as a separate non-default path on 2026-05-20:

```text
poisson_laplace_pirls_sigma_avg
    current retained full-grid path
    -> scalar log-σ candidate grid around current σ
    -> fixed-σ β/u PIRLS refresh for each candidate
    -> diagonal Laplace target weights
    -> weighted β and weighted σ writeback, BLUPs from best candidate

poisson_laplace_pirls_sigma_avg_is
    same path, but grid directions are intercept-vs-slope scales
```

Default prototype settings:

```text
scales = (0.5, 0.75, 1.0, 1.3333333, 2.0)
temperature = 2.0
n_steps = 2
output_mode = beta_sigma
```

The intercept-vs-slope grid uses:

```text
intercept scales = (0.75, 1.0, 1.3333333)
slope scales = (0.5, 1.0, 1.5)
```

First 1000 rows, sequential CPU runs:

| Dataset | part | current FFX | scalar avg FFX | int/slope avg FFX | current σ | scalar avg σ | int/slope avg σ | current BLUP | scalar avg BLUP | int/slope avg BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2214 | 0.2151 | 0.2150 | 0.4529 | 0.4287 | 0.4260 | 0.5245 | 0.5222 | 0.5200 |
| small-p-sampled | valid | 0.2523 | 0.2378 | 0.2370 | 0.4919 | 0.4756 | 0.4649 | 0.5384 | 0.5355 | 0.5325 |
| small-p-sampled | test | 0.2205 | 0.2119 | 0.2121 | 0.4618 | 0.4421 | 0.4353 | 0.5270 | 0.5232 | 0.5208 |
| medium-p-mixed | train | 0.2670 | 0.2438 | 0.2433 | 0.4360 | 0.4198 | 0.4039 | 0.5199 | 0.5129 | 0.5115 |
| medium-p-sampled | valid | 0.3385 | 0.3057 | 0.3062 | 0.5229 | 0.5654 | 0.4910 | 0.6334 | 0.6115 | 0.6078 |
| medium-p-sampled | test | 0.2968 | 0.2831 | 0.2821 | 0.5016 | 0.4723 | 0.4624 | 0.5603 | 0.5560 | 0.5512 |
| large-p-mixed | train | 0.3821 | 0.3436 | 0.3427 | 0.6401 | 0.5536 | 0.5373 | 0.6980 | 0.6472 | 0.6452 |
| large-p-sampled | valid | 0.4422 | 0.4131 | 0.4127 | 0.5475 | 0.5261 | 0.5109 | 0.6414 | 0.6318 | 0.6272 |
| large-p-sampled | test | 0.4608 | 0.4358 | 0.4353 | 0.6215 | 0.5124 | 0.5242 | 0.8082 | 0.7651 | 0.7614 |

Temperature sensitivity on small mixed was weak: β-only FFX was `0.2150` at `T=1`,
`0.2151` at `T=2`, and `0.2153` at `T=4`. Writing back σ from the weighted posterior
improved σ materially on small mixed and improved all reported metrics on medium mixed.

Interpretation:

- Scalar σ averaging improves FFX, σ, and BLUP on every sampled and large row tested
  except medium sampled valid σ, where scalar averaging improves FFX/BLUP but worsens σ.
- Intercept-vs-slope averaging is a small FFX tie/slight win over scalar averaging on most
  rows and usually improves σ/BLUP. The main exception is large sampled test σ, where it
  regresses from `0.5124` to `0.5242` while still improving FFX/BLUP slightly.
- This supports promoting scalar averaging to default. It does not yet prove that the
  richer grid should replace scalar averaging.

Direct Laplace-Target β/u Refinement Prototype
----------------------------------------------

Implemented as a separate non-default path on 2026-05-21:

```text
poisson_laplace_target_refine
    current default
    -> hold σ fixed
    -> take 1-2 small autograd Adam steps on β and BLUP modes
       against the diagonal Laplace target including the logdet curvature term
    -> accept per row only if that Laplace target improves
```

Quick 1k validation:

| Dataset | part | current FFX | target refine FFX | current σ | target refine σ | current BLUP | target refine BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2151 | 0.2131 | 0.4287 | 0.4287 | 0.5222 | 0.5224 |
| small-p-sampled | valid | 0.2378 | 0.2365 | 0.4756 | 0.4756 | 0.5355 | 0.5344 |
| small-p-sampled | test | 0.2119 | 0.2106 | 0.4421 | 0.4421 | 0.5232 | 0.5222 |
| medium-p-mixed | train | 0.2438 | 0.2427 | 0.4198 | 0.4198 | 0.5129 | 0.5128 |

Interpretation:

- The signal is directionally positive but small: `~0.001-0.002` absolute FFX NRMSE gain.
- σ is unchanged by design; BLUPs are flat to slightly better.
- This supports testing the architecture further, but it is not enough to make the
  autograd target refinement default yet.

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

1. **Run targeted diagnostics before adding more gates.**

   Needed diagnostics:

   - row-level INLA FFX/σ/BLUP gap versus `poisson_sigma_average_neff`;
   - best σ scale and effective candidate count distribution by size/partition;
   - Laplace target winner quality versus weighted posterior-mean quality;
   - marginal variance correction size `0.5 * z'Σz`, especially q95/max per row;
   - residual high-FFX rows where σ remains biased after scalar averaging.

   Interpretation rule: diffuse σ weights or large marginal variance corrections support
   more hyperparameter averaging; sharp σ weights with bad β point toward stronger
   Laplace-target β/u optimization or a variational Gaussian update.

2. **Do not spend more effort on scalar diagonal covariance tuning.**

   Current update:

   ```text
   sigma_j^2 <- (sum_g (u_gj^2 + diag(A_g^-1)_j) + nu0 * sigma0_j^2) / (m + nu0)
   ```

   The sanity sweep found only small FFX gains from extra final fixed-σ PIRLS steps and no
   convincing σ/BLUP improvement. Leave defaults unchanged for now; `final=4` or `final=6`
   can be retested later if a larger architecture change makes convergence the bottleneck.

3. **Do not promote full-Σ PIRLS without a stronger scoring/averaging change.**

   Full covariance alone was stable but worse than the retained diagonal full-grid path on
   small and medium mixed rows. Keep it as a prototype/diagnostic path. The next serious
   improvement should change the approximation target, not only the covariance
   parameterization.

4. **Keep one accept/reject target.**

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

5. **Continue hyperparameter averaging, but avoid more scalar gate tuning.**

   Scalar averaging is the default. Intercept-vs-slope averaging gives slightly better FFX
   and usually better σ/BLUP, but the gain is modest and not uniform. Next ablations:

   - validate 2-step direct Laplace-target β/u refinement on medium/large sampled rows;
   - compare `output_mode=beta`, `beta_best`, and `beta_sigma` for the intercept-vs-slope
     grid on medium/large sampled rows;
   - if direct target refinement remains positive, test moving it inside each σ candidate
     rather than only after weighted σ output.

6. **Run huge-row validation before 8k.**

   Full 8k Poisson benchmarks remain postponed until one more meaningful accuracy patch or
   complete huge-row validation lands.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods raw poisson_eb poisson_marginal_beta \
    poisson_laplace_pirls_diag poisson_laplace_pirls_full_beta \
    poisson_laplace_pirls_beta current poisson_laplace_pirls_sigma_avg \
    poisson_laplace_pirls_sigma_avg_is poisson_laplace_target_refine \
    --sizes small medium large \
    --batch-size 32 --max-datasets 1000

uv run python -u experiments/analytical/glmm_poisson_pirls_grid_diagnostic.py \
    --sizes small medium --max-datasets 1000 --batch-size 32

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
