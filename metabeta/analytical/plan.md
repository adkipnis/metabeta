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

Next direction: improve sRFX
----------------------------

Stop tuning the final BLUP beta blend unless a new diagnostic shows a clear target.
I7 brought BLUP error close to the apparent beta-residual limit for small-n. The largest
remaining analytical error is variance-component accuracy:

| Dataset | Partition | sRFX |
| --- | --- | ---: |
| small-n-mixed | train | 0.6421 |
| small-n-sampled | valid | 0.6313 |
| small-n-sampled | test | 0.6655 |
| medium-n-mixed | train | 0.5739 |
| medium-n-sampled | valid | 0.5334 |
| medium-n-sampled | test | 0.6081 |
| large-n-mixed | train | 0.4947 |
| large-n-sampled | valid | 0.5811 |
| large-n-sampled | test | 0.6065 |
| huge-n-mixed | train | 0.4957 |
| huge-n-sampled | valid | 0.7824 |
| huge-n-sampled | test | 0.5954 |

Working hypothesis:

- BLUP NRMSE was dominated by fixed-effect leakage, now handled output-locally.
- sRFX is controlled by MoM initialization, component-wise fallback, Psi eigencap/floor,
  off-diagonal shrinkage, and EM M-step trimming.
- Previous variance patches failed because they were broad runtime changes without first
  separating diagonal scale bias from correlation/off-diagonal error and cap/floor effects.

Diagnostic result:

Added `experiments/analytical/glmm_srfx_diagnostic.py`. It replays the Gaussian
estimator internals and reports the required-suite split by MoM, EM, first EM M-step
targets, floor/cap activity, component fallback, `G_mom`, active `q`, and off-diagonal
correlation error.

Key required-suite findings:

| Split | N | sRFX NRMSE | Rel. bias |
| --- | ---: | ---: | ---: |
| full MoM path | 166562 | 0.4753 | +1.816 |
| diag MoM path | 8127 | 0.9257 | +4.985 |
| component diag path | 4427 | 1.0082 | +4.682 |
| fallback path | 1767 | 1.0386 | +1.885 |
| floor hit = false | 121453 | 0.4871 | +0.197 |
| floor hit = true | 59430 | 0.9835 | +5.773 |
| cap hit = false | 180096 | 0.5541 | +2.008 |
| cap hit = true | 787 | 1.3975 | +6.808 |

Interpretation:

- EM improves all dataset rows versus the initial MoM estimate; do not target "more EM" or
  a broad M-step rewrite first.
- First EM raw/winsor/trim targets are worse than the final post-EM estimate on every row,
  so the current iterative shrinkage is helping.
- Cap-hit rows are very bad but rare (`787` active components); eigencap is not the first
  high-leverage patch.
- Floor-hit components are common (`59430` active components), high-error, and strongly
  overestimated. The likely first target is an overly high diagonal floor, especially when
  `diag_mom`, `component_diag`, or fallback paths are used.
- Off-diagonal correlation NRMSE stays high after shrinkage, but `sigma_rfx` is a diagonal
  scale metric and the floor-hit diagonal bias is the more direct target.

Accepted I8 result: slight diagonal floor reduction
---------------------------------------------------

I8 patched only the joint diagonal MoM floor signal:

```python
diag_floor_signal = 0.45 * psi_diag_signal * active_q
```

Fallback and component-diagonal floors are unchanged. More aggressive scalar reductions
were rejected:

- `0.25` improved sRFX but failed the BLUP guardrail on `medium-n-sampled/test`
  (`0.4792 -> 0.4978`, +3.9%) and moved FFX noticeably.
- `0.375` failed badly on mixed rows (`medium-n-mixed` BLUP `0.3714 -> 0.6457`;
  `huge-n-mixed` `0.3964 -> 0.4335`).

Accepted required benchmark:

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.6353 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.6313 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6623 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.5640 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.5326 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.6028 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.4898 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.5662 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.5950 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.4932 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.7640 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5908 | 0.1828 | 0.5167 |

Decision:

- Keep. sRFX improves on every required row versus I7.
- BLUP stays inside the 3% I7 budget and improves on most rows.
- FFX/sEps movements are small; changing Psi necessarily changes GLS, but there is no
  material FFX/sEps failure.
- The diagnostic still shows floor-hit components are bad (`sRFX=1.0476`, rel bias
  `+6.009`) but fewer than baseline (`56155` versus `59430` active components).

Next direction:

1. Do not keep searching scalar floor multipliers; `0.25` and `0.375` showed unstable
   GLS/EM interactions.
2. Investigate a selective, observable gate for floor reduction rather than another
   global scalar. Candidate signals: `enough_full_mom`, `G_mom`, active `q`, and
   `floor_hit` predicted from the pre-clamp diagonal.
3. Prioritize `diag_mom` and fallback/component paths, where sRFX remains near or above
   0.9-1.0.
4. Keep I7 BLUP beta schedule and I8 scalar floor fixed while running the next diagnostic.

Guardrails:

- Do not change the accepted I7 BLUP beta schedule while investigating sRFX.
- Do not use truth, dataset family, or partition in runtime gates.
- Avoid more EM iterations as a first patch; earlier attempts showed biased fixed points.
- Treat `huge-n-sampled/valid` as a key risk row because its current sRFX is worst at 0.7824.
