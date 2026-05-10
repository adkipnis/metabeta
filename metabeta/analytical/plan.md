Analytical GLMM Plan
====================

Last updated: 2026-05-10.

Current Decision
----------------

Stop the current GLMM analytical patch series here. The implementation is stable
enough for handoff, and the remaining obvious directions are more likely to produce
small output-calibration gains than meaningful estimator improvements.

Current Implementation
----------------------

- BLUP point estimates use the Gaussian full path plus an output-local beta blend
  for final BLUP residuals:
  - active d <= 4: pooled OLS residuals.
  - active d 5-8: 65% pooled OLS, 35% GLS.
  - active d > 8: 75% pooled OLS, 25% GLS.
- Runtime Psi uses the current MoM/EM path with conservative floors retained for
  stability.
- Reported marginal sRFX uses output-only floor-hit calibration:
  - q = 1 and low floor-ratio: sigma factor `0.70`.
  - q = 2 and low floor-ratio: sigma factor `0.85`.
  - q > 2 floor-hit: sigma factor `0.55`.
- `Psi` returned by the estimator receives the same diagonal output calibration as
  `sigma_rfx_est`. BLUP, FFX, and sEps are computed before this calibration.
- `_remlNewtonStep` remains in `normal.py` but is not wired in.

Final Benchmark
---------------

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.6100 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.6291 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6507 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.4501 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.5302 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.5979 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.4614 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.5385 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.5264 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.4450 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.6535 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5685 | 0.1828 | 0.5167 |

What Worked
-----------

- Output-local changes were safest: BLUP beta blending and sRFX calibration improved
  target metrics without moving unrelated public estimates.
- Component-count gating and EM BLUP winsorization fixed major high-q BLUP/FFX
  failures by preventing noisy components and extreme groups from dominating Psi.
- Mild floor tuning helped sRFX only when it stayed conservative.

What Failed
-----------

- Broad runtime floor reductions moved FFX, sEps, or BLUP.
- More EM, REML gates, and psi_df/G_mom iteration gates found biased fixed points in
  some regimes.
- Beta changes that affected reported `beta_est` traded BLUP gains for FFX
  regressions.
- Broad weak-path output schedules were too blunt; clean wins came only from q and
  floor-ratio gates.

Remaining Weaknesses
--------------------

- sRFX is still weakest on sampled rows, especially huge valid and small test.
- Floor-hit Psi components remain the main source of marginal scale bias.
- Further analytical gains are likely small unless the estimator is redesigned
  rather than patched.

Handoff Commands
----------------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/normal.py experiments/analytical
```
