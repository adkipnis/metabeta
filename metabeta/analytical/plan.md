Plan
====

Last updated: 2026-05-09, after I7 active-d adaptive alpha benchmark.

Current estimator state
-----------------------

Implemented and kept:

- Fix 1: add Psi/G_mom to `blup_var` as a dataset-level variance floor.
- Fix 2: loosen delta-method denominator cap from 4 to 2.
- Fix 4: shrink off-diagonal Psi in the Normal path.
- Fix 5: keep batch-max EM convergence early exit, with max EM iterations still 5.
- Fix 7: use a per-component count floor in component-wise MoM.
- Fix 9: adaptive M-step BLUP winsorization at 10x current random-effect scale.
- Fix C: trim outlier BLUP outer products in the EM Psi M-step.
- I5: compute final BLUP residuals with a BLUP-only beta blend.
- I6: tune that blend to alpha=0.75:
  `beta_for_blup = 0.25 * beta_gls + 0.75 * beta_ols`.
- I7: make the BLUP-only beta blend active-d adaptive:
  alpha 1.00 for active d <= 4, 0.65 for active d 5-8, and 0.75 above that.

Also present but unused:

- `_remlNewtonStep` in `metabeta/analytical/normal.py`.

Active problem
--------------

The remaining priority is BLUP accuracy without spending the medium/sampled regression budget.

Fix C baseline:

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2249 | 0.6421 | 0.0839 | 1.0687 |
| medium-n-mixed | train | 0.1452 | 0.5739 | 0.0671 | 0.3749 |
| large-n-mixed | train | 0.2686 | 0.4947 | 0.0724 | 0.3670 |
| huge-n-mixed | train | 0.3034 | 0.4957 | 0.0648 | 0.4663 |
| small-n-sampled | valid | 0.1551 | 0.6313 | 0.1031 | 0.7044 |
| small-n-sampled | test | 0.1686 | 0.6655 | 0.1002 | 0.7898 |
| medium-n-sampled | valid | 0.3625 | 0.5334 | 0.0978 | 0.5566 |
| medium-n-sampled | test | 0.2437 | 0.6081 | 0.1029 | 0.5367 |
| large-n-sampled | valid | 0.3874 | 0.5811 | 0.1104 | 0.6642 |
| large-n-sampled | test | 0.4959 | 0.6065 | 0.1078 | 0.6774 |
| huge-n-sampled | valid | 0.4208 | 0.7824 | 0.1643 | 0.5827 |
| huge-n-sampled | test | 0.4662 | 0.5954 | 0.1826 | 0.6631 |

Previous accepted I5 alpha=0.50 results:

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2249 | 0.6421 | 0.0839 | 0.7198 |
| medium-n-mixed | train | 0.1452 | 0.5739 | 0.0671 | 0.3660 |
| large-n-mixed | train | 0.2686 | 0.4947 | 0.0724 | 0.3605 |
| huge-n-mixed | train | 0.3034 | 0.4957 | 0.0648 | 0.4038 |
| small-n-sampled | valid | 0.1551 | 0.6313 | 0.1031 | 0.5016 |
| small-n-sampled | test | 0.1686 | 0.6655 | 0.1002 | 0.5319 |
| medium-n-sampled | valid | 0.3625 | 0.5334 | 0.0978 | 0.5201 |
| medium-n-sampled | test | 0.2437 | 0.6081 | 0.1029 | 0.4858 |
| large-n-sampled | valid | 0.3874 | 0.5811 | 0.1104 | 0.5163 |
| large-n-sampled | test | 0.4959 | 0.6065 | 0.1078 | 0.5312 |
| huge-n-sampled | valid | 0.4208 | 0.7824 | 0.1643 | 0.5163 |
| huge-n-sampled | test | 0.4662 | 0.5954 | 0.1826 | 0.5423 |

Working diagnosis
-----------------

- I4 showed that small-n BLUP error is mostly fixed-effect leakage into the random-effect
  residual, not post-EM Psi/sigma collapse.
- True Psi/sigma with `beta_hat` did not solve P1: BLUP 1.0687 -> 1.0599.
- True beta nearly solved it: BLUP 1.0687 -> 0.2931.
- `beta_wg` is not a viable fallback: BLUP 3.33-3.36 in oracle ablations.
- Pooled OLS is a useful residual target for BLUPs, but changing reported `beta_est`
  regressed FFX. Keep beta changes output-local unless diagnostics say otherwise.

Lessons to preserve
-------------------

- Fix upstream estimator errors before downstream calibration. Fix 7 mattered more than
  variance floors because it corrected component-wise Psi initialization.
- More EM iterations can converge to a biased fixed point. The 5-iteration cap is useful
  regularization until the M-step bias is solved.
- Count-based gates such as G_mom and psi_df do not identify fixed-point quality.
- Low post-EM Psi/sigma ratio rows were already easy cases. Do not revive the low-ratio
  REML gate without new evidence.
- Avoid dataset-family gates and truth-based projection gates. Runtime gates must use
  observable numerical diagnostics only.

Rejected paths
--------------

| Attempt | Result |
| --- | --- |
| Fix A: use `beta_wg` in MoM residuals | Reverted. Medium-n-mixed FFX blew up to ~80. |
| Fix B: psi_df-gated EM extension | Reverted. Medium-n-mixed BLUP 0.3748 -> 0.9601. |
| Fix D: regularize EM sigma2 toward Stage 1 | Reverted. Huge-n-mixed BLUP +31%. |
| Fix E: refresh mom4 mask after first EM iteration | No-op. Group diagnostics are structural. |
| I3 Option C/D/E REML gates | Reverted or abandoned. No BLUP gain with large sRFX risk. |
| Final GLS ridge 1e-6 -> 1e-5 | Reverted. No small-n gain; huge-n-mixed BLUP regressed. |
| Reported-beta blend with beta_ols | Reverted. BLUPs improved but FFX regressed. |

Benchmark protocol
------------------

For each patch, run:

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
```

The required suite is:

- mixed datasets: `small|medium|large|huge-n-mixed`, train epochs 1 and 2.
- sampled datasets: `small|medium|large|huge-n-sampled`, valid and test.

General keep rules:

- FFX, sRFX, and sEps should remain unchanged unless the patch intentionally targets them.
- No required BLUP row may regress by more than 3% versus the relevant baseline.
- `small-n-mixed` must improve materially, not just move within noise.
- Revert immediately after a clear medium/large/huge mixed regression.

Accepted I6 result: beta blend alpha=0.75
-----------------------------------------

I6-1 changed only the final BLUP residual beta:

```python
beta_for_blup = 0.25 * beta_gls + 0.75 * beta_ols
```

Reported `beta_est`, Psi, sigma2, `beta_var`, and `blup_var` are unchanged.

Required benchmark:

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2249 | 0.6421 | 0.0839 | 0.5355 |
| small-n-sampled | valid | 0.1551 | 0.6313 | 0.1031 | 0.4409 |
| small-n-sampled | test | 0.1686 | 0.6655 | 0.1002 | 0.4480 |
| medium-n-mixed | train | 0.1452 | 0.5739 | 0.0671 | 0.3769 |
| medium-n-sampled | valid | 0.3625 | 0.5334 | 0.0978 | 0.4796 |
| medium-n-sampled | test | 0.2437 | 0.6081 | 0.1029 | 0.4773 |
| large-n-mixed | train | 0.2686 | 0.4947 | 0.0724 | 0.3643 |
| large-n-sampled | valid | 0.3874 | 0.5811 | 0.1104 | 0.4782 |
| large-n-sampled | test | 0.4959 | 0.6065 | 0.1078 | 0.4945 |
| huge-n-mixed | train | 0.3034 | 0.4957 | 0.0648 | 0.3964 |
| huge-n-sampled | valid | 0.4208 | 0.7824 | 0.1643 | 0.5095 |
| huge-n-sampled | test | 0.4662 | 0.5954 | 0.1826 | 0.5186 |

Decision:

- Keep. `small-n-mixed` improves from 0.7198 to 0.5355, a 25.6% gain.
- FFX, sRFX, and sEps are unchanged across the required suite.
- No BLUP row regresses by more than 3% versus Fix C.
- `medium-n-mixed` is the limiting case: 0.3769 versus the current-I5 guardrail of
  `0.3660 * 1.03 = 0.3770`.

Accepted I7 result: active-d adaptive alpha
-------------------------------------------

I7 used `glmm_alpha_gate_diagnostic.py` to compare scalar alphas and observable gates.
Simple beta-mask/rank gates were not sufficient: they either lost too much small-n gain or
regressed medium rows. The accepted schedule uses active fixed-effect dimension inferred
from the design matrix:

```python
alpha = 1.00  if active_d <= 4
alpha = 0.65  if 5 <= active_d <= 8
alpha = 0.75  otherwise
```

Required benchmark:

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2249 | 0.6421 | 0.0839 | 0.3637 |
| small-n-sampled | valid | 0.1551 | 0.6313 | 0.1031 | 0.4248 |
| small-n-sampled | test | 0.1686 | 0.6655 | 0.1002 | 0.4209 |
| medium-n-mixed | train | 0.1452 | 0.5739 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.5334 | 0.0978 | 0.4922 |
| medium-n-sampled | test | 0.2437 | 0.6081 | 0.1029 | 0.4792 |
| large-n-mixed | train | 0.2686 | 0.4947 | 0.0724 | 0.3643 |
| large-n-sampled | valid | 0.3874 | 0.5811 | 0.1104 | 0.4782 |
| large-n-sampled | test | 0.4959 | 0.6065 | 0.1078 | 0.4945 |
| huge-n-mixed | train | 0.3034 | 0.4957 | 0.0648 | 0.3964 |
| huge-n-sampled | valid | 0.4208 | 0.7824 | 0.1643 | 0.5095 |
| huge-n-sampled | test | 0.4662 | 0.5954 | 0.1826 | 0.5186 |

Decision:

- Keep. `small-n-mixed` improves from 0.5355 to 0.3637 versus I6.
- `medium-n-mixed` moves away from the I5 guardrail: 0.3769 -> 0.3714.
- FFX, sRFX, and sEps are unchanged across the required suite.
- The largest BLUP regression versus I6 is `medium-n-sampled/valid`: 0.4796 -> 0.4922
  (+2.6%), inside the 3% budget.

Next direction
--------------

Do not keep tuning scalar alpha without a new diagnostic. The current BLUP error is close
to the apparent beta-residual limit for small-n, and remaining risk is concentrated in
sampled medium/high-d rows.

Next investigation should target one of:

1. Better observable gate for medium sampled rows, using design diagnostics beyond active d
   only if it improves `medium-n-sampled/valid` without losing the small-n gain.
2. Variance-component accuracy (`sRFX`) separately from BLUP residual beta, since sRFX remains
   much larger than BLUP error on several rows.
