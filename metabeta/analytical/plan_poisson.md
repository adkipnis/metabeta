Poisson GLMM Plan
=================

Last updated: 2026-05-21 (variational Gaussian diagnostic + σ-blend patch)

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
- `poisson_variational_gaussian`: default plus Gaussian variational posterior-mean update
  with conservative σ blending; current leading prototype, not default.
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

### Gaussian Variational Posterior-Mean Update

`poisson_variational_gaussian` starts from the default path, then updates
`q(u_g)=N(m_g,V_g)` using the expected Poisson mean
`exp(Xβ + Zm_g + 0.5 * diag(Z V_g Z'))`. Diagonal σ is updated from
`m_g² + diag(V_g)`, blended conservatively with the previous σ estimate
(`sigma_blend=0.25`). The candidate is accepted only if its ELBO-style target improves.

First 1000 rows, sequential CPU run after the conservative σ-blend patch.

| Dataset | part | default FFX | var-Gauss FFX | default σ | var-Gauss σ | default BLUP | var-Gauss BLUP | var-Gauss ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.2151 | 0.2129 | 0.4287 | 0.4289 | 0.5222 | 0.5208 | 39.3 |
| small-p-sampled | valid | 0.2378 | 0.2336 | 0.4756 | 0.4779 | 0.5355 | 0.5330 | 36.3 |
| small-p-sampled | test | 0.2119 | 0.2111 | 0.4421 | 0.4407 | 0.5232 | 0.5211 | 36.7 |
| medium-p-mixed | train | 0.2438 | 0.2314 | 0.4198 | 0.4086 | 0.5129 | 0.5084 | 70.3 |
| medium-p-sampled | valid | 0.3057 | 0.2753 | 0.5654 | 0.5315 | 0.6115 | 0.5901 | 83.3 |
| medium-p-sampled | test | 0.2831 | 0.2745 | 0.4723 | 0.4570 | 0.5560 | 0.5519 | 77.1 |
| large-p-mixed | train | 0.3436 | 0.3074 | 0.5536 | 0.5315 | 0.6472 | 0.5938 | 84.4 |
| large-p-sampled | valid | 0.4131 | 0.3843 | 0.5261 | 0.5112 | 0.6318 | 0.6261 | 77.7 |
| large-p-sampled | test | 0.4358 | 0.4130 | 0.5124 | 0.5195 | 0.7651 | 0.7286 | 89.1 |

Interpretation:

- This is the first Poisson prototype that improves FFX on every tested small/medium/large
  row while usually improving σ and BLUP too.
- Gains are materially larger than direct fixed-σ Laplace target refinement, especially on
  medium/large rows.
- The conservative σ blend improves FFX on every row and usually improves σ/BLUP.
- The only remaining σ regression in this table is `large-p-sampled test`; it is much
  smaller than with the earlier `sigma_blend=0.5` prototype.
- Treat this as the leading architecture to tune or extend, but not as the default until
  sampled/huge rows and σ writeback behavior are better understood.

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
- Do not make β-only output the default yet; keep it as the fallback idea if future rows
  show unacceptable σ/BLUP regressions.
- The next likely useful guard is targeted σ writeback suppression for extreme marginal
  variance rows, not more generic gate tuning.

Most Promising Architectural Change
-----------------------------------

The most plausible route to FFX around `~0.20` remains a posterior-mean-oriented
Poisson variational/Laplace update, not another posthoc correction around the current
conditional mode. The first `poisson_variational_gaussian` prototype validates the
direction, but the gap is still too large on large rows.

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

Next refinements for this path:

- Add a principled σ-writeback guard only for extreme rows where VG improves β but the
  random-effect marginal correction is already very large or sharply selected.
- Test one richer posterior-mean update: scalar/intercept-vs-slope σ averaging around the
  variational candidate rather than around the conditional-mode candidate.
- If large-row FFX remains far from INLA after that, prototype a more ELBO-consistent
  update of `V_g` instead of setting it only from local curvature after each mean step.

Secondary direction:

- Move direct Laplace-target β/u refinement inside each σ candidate before weighting. This
  is smaller than the variational path and may improve scalar averaging, but current
  fixed-σ gains suggest it is unlikely to close the large-row gap by itself.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_variational_gaussian \
    --sizes small medium large \
    --batch-size 32 --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
