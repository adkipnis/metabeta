Analytical GLMM Plan
====================

Last updated: 2026-05-10.

Current Decision
----------------

The current production GLMM analytical baseline is stable. Only two non-MCMC
directions remain worth testing:

1. refined marginal MAP / REML over variance components;
2. Laplace curvature and uncertainty around the MAP optimum.

Option 1 is the recommended next diagnostic. It should stay experiment-only until
it beats the integrated `marg20_hyb_srfx` path on the required suite without moving
FFX, sEps, or BLUP beyond noise-level changes.

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
- The current production MAP path replaces only `sigma_rfx_est` and the `Psi`
  diagonal. FFX, sEps, and BLUP remain MoM/EM-derived.
- `_remlNewtonStep` remains in `normal.py` but is not wired in.

Final Benchmark
---------------

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.5650 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.5819 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6242 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.3700 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.4671 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.5129 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.3849 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.4679 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.4677 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.3932 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.5911 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5050 | 0.1828 | 0.5167 |

Future Directions
-----------------

### Option 1: REML/Profile Marginal MAP Diagnostic

Run `experiments/analytical/glmm_reml_diagnostic.py` before changing production
code. Compare:

- `current`: production `glmm()` with default marginal MAP sRFX refinement.
- `mom_em`: historical MoM/EM baseline via `glmm(..., map_refine=False)`.
- `reml_diag`: fixed beta, fixed correlation, fixed sigma(Eps), optimize only
  `log_sigma_rfx`.
- `reml_diag_seps`: fixed beta and fixed correlation, optimize `log_sigma_rfx`
  and `log_sigma_eps`.

Use MoM/EM output as initialization and the exact Gaussian marginal likelihood
from `metabeta/analytical/map.py` as the objective base. Defaults are 20 Adam
steps, learning rate 0.03, gradient clipping at 10.0, and sigma clamps in
`[1e-4, 20.0]`. If the objective or refined values become non-finite, keep the
current production stats for that batch.

Acceptance bar: improve sRFX in most required-suite cells, especially sampled
rows, while FFX, sEps, and BLUP do not regress materially. Keep the result as a
diagnostic if gains are output-local only.

### Option 2: Laplace Uncertainty Around MAP

Use curvature around the MAP optimum to expose uncertainty/context features, not
as the primary route to point-estimate gains. Expected cost is higher than the
current output-only MAP path because curvature must be estimated after the MAP
optimization; expected gain is better uncertainty information rather than lower
required-suite point NRMSE.

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
- VI, SGLD, SVGD, MH, and full MCMC are not active analytical-estimator plans for
  this repo.

Handoff Commands
----------------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_reml_diagnostic.py
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
