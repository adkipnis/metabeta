Poisson GLMM Plan
=================

Last updated: 2026-05-21 (debloated after scalar σ averaging default)

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

The final scalar averaging pass approximates local hyperparameter integration over
diagonal σ scales and writes back weighted β/σ plus BLUPs from the best-scoring candidate.
This is now the default because it improved FFX on every sampled and large row tested.

Important method names:

- `current` / `default`: current retained path with scalar σ averaging.
- `poisson_laplace_pirls_sigma_avg`: explicit scalar σ averaging path.
- `poisson_laplace_pirls_sigma_avg_is`: intercept-vs-slope σ averaging prototype.
- `poisson_laplace_target_refine`: default plus 1-2 direct β/u autograd steps under the
  diagonal Laplace target; promising but not default.
- `poisson_laplace_pirls_full` / `poisson_laplace_pirls_full_beta`: full-Σ prototypes;
  stable but worse than the diagonal default in the tested rows.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU runs on 2026-05-20/21. Lower NRMSE is better.
INLA values are the current first-1000 diagonal R-INLA references.

| Dataset | part | default FFX | INLA FFX | FFX gap | default σ | INLA σ | default BLUP | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2151 | 0.1835 | +0.0316 | 0.4287 | 0.3404 | 0.5222 | 0.4936 | 33.5 |
| small-p-sampled | valid | 0.2378 | 0.2276 | +0.0102 | 0.4756 | 0.4356 | 0.5355 | 0.5309 | 36.7 |
| small-p-sampled | test | 0.2119 | 0.1997 | +0.0122 | 0.4421 | 0.3966 | 0.5232 | 0.5281 | 33.8 |
| medium-p-mixed | train | 0.2438 | 0.1675 | +0.0763 | 0.4198 | 0.3214 | 0.5129 | 0.4789 | 64.1 |
| medium-p-sampled | valid | 0.3057 | 0.2146 | +0.0911 | 0.5654 | 0.4209 | 0.6115 | 0.5618 | 77.6 |
| medium-p-sampled | test | 0.2831 | 0.2267 | +0.0564 | 0.4723 | 0.3883 | 0.5560 | 0.5849 | 71.6 |
| large-p-mixed | train | 0.3436 | 0.1778 | +0.1658 | 0.5536 | 0.3076 | 0.6472 | 0.5001 | 82.4 |
| large-p-sampled | valid | 0.4131 | 0.2467 | +0.1664 | 0.5261 | 0.4232 | 0.6318 | 0.5870 | 85.2 |
| large-p-sampled | test | 0.4358 | 0.2186 | +0.2172 | 0.5124 | 0.3439 | 0.7651 | 0.5618 | 86.7 |

Interpretation:

- Small rows are near INLA for FFX, but σ remains worse.
- Medium and large rows are still substantially behind INLA, especially for FFX and σ.
- The remaining FFX gap is likely driven by imperfect σ/hyperparameter integration through
  the Poisson log-link marginal mean, not by a missing scalar gate.
- Runtime is already near the intended range, so the next serious patch can spend more
  compute if it materially reduces FFX.

What We Retired
---------------

- β-only σ grid: replaced by full-candidate grid and scalar weighted σ averaging.
- β-only AGQ optimizer: slower and did not close the main gap.
- More gate tuning around the old EB/grid path: low expected value.
- Broad σ inflation grids: rare but large σ outliers.
- Full-Σ PIRLS as default: stable and similarly fast, but worse than the diagonal default.
- More scalar covariance-update tuning: only small gains and did not improve σ/BLUP enough.

Active Prototypes
-----------------

### Intercept-vs-Slope σ Averaging

This usually ties or slightly beats scalar σ averaging for FFX and often improves σ/BLUP,
but gains are small and not uniform. Keep it as an ablation until diagnostics show a clear
case where separate intercept/slope hyperparameter uncertainty matters.

### Direct Laplace-Target β/u Refinement

`poisson_laplace_target_refine` holds σ fixed and takes small autograd Adam steps on β/u
against the diagonal Laplace target, accepting per row only if the target improves.

Quick 1k validation:

| Dataset | part | default FFX | target refine FFX | default σ | target refine σ | default BLUP | target refine BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2151 | 0.2131 | 0.4287 | 0.4287 | 0.5222 | 0.5224 |
| small-p-sampled | valid | 0.2378 | 0.2365 | 0.4756 | 0.4756 | 0.5355 | 0.5344 |
| small-p-sampled | test | 0.2119 | 0.2106 | 0.4421 | 0.4421 | 0.5232 | 0.5222 |
| medium-p-mixed | train | 0.2438 | 0.2427 | 0.4198 | 0.4198 | 0.5129 | 0.5128 |

The signal is positive but too small to close the INLA gap. This argues against making
posthoc fixed-σ β/u refinement the main architecture.

Next Diagnostics
----------------

Run these before adding more gates:

- Row-level INLA FFX/σ/BLUP gap versus `poisson_sigma_average_neff`.
- Best σ scale and effective candidate count distribution by size/partition.
- Laplace target winner quality versus weighted posterior-mean quality.
- Marginal variance correction size `0.5 * z'Σz`, especially q95/max per row.
- Residual high-FFX rows where σ remains biased after scalar averaging.
- Whether large-row failures are concentrated in high `d`, high `q`, high marginal
  variance correction, diffuse σ weights, or sharp-but-wrong σ weights.

Interpretation rule:

- Diffuse σ weights or large marginal variance corrections support richer hyperparameter
  averaging.
- Sharp σ weights with bad β point toward a better objective/optimization target.
- Persistently biased σ after averaging points toward changing the approximation class,
  not retuning the current diagonal EB update.

Most Promising Architectural Change
-----------------------------------

The most plausible route to FFX around `~0.20` is a posterior-mean-oriented Poisson
Laplace/variational Gaussian update, not another posthoc correction around the current
conditional mode.

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

- INLA likely wins by posterior/hyperparameter averaging, while our current path is still
  mostly a corrected mode.
- The Poisson log link makes β highly sensitive to uncertainty in random effects through
  `exp(0.5 * z'Σz)`.
- Scalar σ averaging helps, which is direct evidence that posterior averaging is useful.
- Full-Σ covariance did not help, so the missing piece is more likely uncertainty
  propagation than covariance shape.
- Direct fixed-σ Laplace target refinement helps only marginally, so optimizing the same
  conditional mode harder is unlikely to reach INLA territory.

This should be tested as a separate path first, with diagonal `V_g` or full `q x q` `V_g`
depending on implementation cost. For `q <= 5`, full `V_g` is computationally feasible;
the main risk is stability and target calibration, not FLOPs.

Secondary direction:

- Move direct Laplace-target β/u refinement inside each σ candidate before weighting. This
  is smaller than the variational path and may improve scalar averaging, but current
  fixed-σ gains suggest it is unlikely to close the large-row gap by itself.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_laplace_target_refine \
    --poisson-laplace-target-refine-steps 2 \
    --sizes small medium large \
    --batch-size 32 --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
