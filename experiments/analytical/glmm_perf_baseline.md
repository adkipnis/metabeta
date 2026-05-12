GLMM Analytical Estimator Summary
=================================

Last updated: 2026-05-11 after adding the diagonal-MAP final GLS/BLUP pass.

This file records the current retained Gaussian GLMM analytical benchmark and the
closed REML diagnostic decision. Historical MAP/REML sweep scripts were removed;
the reusable scripts are documented in `README.md`.

Current Production MAP Benchmark
--------------------------------

Command:

```bash
uv run python experiments/analytical/glmm_required_benchmark.py --methods current raw
```

Required suite: mixed train epochs 1-2 and sampled valid/test for
`small|medium|large|huge`. Current production MAP reports MAP `sigma_rfx_est`,
writes a diagonal MAP `Psi`, and recomputes final Gaussian beta/BLUPs with that
diagonal covariance. This keeps the useful MAP variance scale while excluding
noisy estimated correlations from final BLUP shrinkage. Reported sigma(Eps) still
comes from the within-group projection estimate. The existing calibrated
`blup_var` output is preserved pending a separate variance-calibration pass.

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1694 | 0.3828 | 0.1658 | 0.3795 |
| small-n-sampled | valid | 0.1983 | 0.4917 | 0.2147 | 0.4675 |
| small-n-sampled | test | 0.1963 | 0.4344 | 0.1973 | 0.4202 |
| medium-n-mixed | train | 0.5100 | 0.5176 | 0.1294 | 0.5356 |
| medium-n-sampled | valid | 0.3682 | 0.4144 | 0.1853 | 0.4854 |
| medium-n-sampled | test | 0.4830 | 0.4496 | 0.1755 | 0.4835 |
| large-n-mixed | train | 0.9226 | 0.3648 | 0.1117 | 0.4215 |
| large-n-sampled | valid | 0.9175 | 0.6349 | 0.1344 | 0.5460 |
| large-n-sampled | test | 0.7292 | 0.4015 | 0.1329 | 0.4954 |
| huge-n-mixed | train | 0.9431 | 0.5548 | 0.0942 | 0.4549 |
| huge-n-sampled | valid | 1.1068 | 0.4590 | 0.1197 | 0.5050 |
| huge-n-sampled | test | 1.3276 | 0.4084 | 0.1171 | 0.4854 |

Raw Estimator Pass
------------------

Retained changes from the first raw-estimator pass:

- Low-dimensional final BLUP residuals now use the same 65% pooled-OLS beta blend
  as the medium-dimensional branch instead of a 100% pooled-OLS residual.
- The final GLS/BLUP pass and reported sigma(Eps) now use the within-group
  projection sigma(Eps). This fixed the large/huge sigma(Eps) outliers and slightly
  improved sampled BLUPs, with a medium-mixed FFX tradeoff to monitor.

Rejected candidate:

- Lowering the high-dimensional BLUP beta blend from 0.75 to 0.65/0.50 had mixed
  results and regressed large-valid or huge-mixed BLUPs, so the high-dimensional
  branch remains 0.75.
- The refreshed raw baseline now remains the non-MAP fallback. Current MAP improves
  sigma(RFX) in every required cell and, with the diagonal-MAP final covariance,
  improves global BLUP from 0.4978 to 0.4682.

MAP Diagonal BLUP Pass
----------------------

The raw-stage attribution diagnostic showed that the BLUP ceiling is mostly the
diagonal variance scale, not off-diagonal correlation:

| Method | FFX | sRFX | sEps | BLUP |
| --- | ---: | ---: | ---: | ---: |
| legacy output-local MAP | 0.6696 | 0.4585 | 0.1331 | 0.4978 |
| output Psi recompute | 0.8141 | 0.7103 | 0.1331 | 0.5341 |
| MAP diagonal recompute | 0.6624 | 0.4585 | 0.1331 | 0.4682 |
| oracle Psi diagonal | 0.6679 | 0.0000 | 0.1331 | 0.4408 |
| oracle full Psi | 0.6673 | 0.0000 | 0.1331 | 0.4301 |

The production path therefore uses the MAP sigma(RFX) diagonal as the final
Gaussian covariance for beta/BLUP recompute. The legacy output-local behavior is
still available for diagnostics via `glmm(..., map_recompute_blup=False)`.

Retired REML Diagnostics
------------------------

REML/profile-MAP was tested as an experiment-only variance-scale refinement and
retired after the refreshed benchmark. The diagnostic script and package-level
REML support were removed; MAP is the only retained production path.

Full-suite row-weighted sRFX NRMSE:

| Method | sRFX |
| --- | ---: |
| mom_em | 0.7103 |
| current MAP | 0.4585 |
| raw-initialized REML | 0.4612 |
| gated REML | 0.4608 |
| REML with sigma(Eps) optimized | 0.4608 |
| current-initialized REML | 0.4738 |

Required-suite cells:

| Dataset | Partition | current MAP | gated REML |
| --- | --- | ---: | ---: |
| small-n-mixed | train | 0.3828 | 0.3805 |
| small-n-sampled | valid | 0.4917 | 0.4944 |
| small-n-sampled | test | 0.4344 | 0.4367 |
| medium-n-mixed | train | 0.5176 | 0.5326 |
| medium-n-sampled | valid | 0.4144 | 0.4171 |
| medium-n-sampled | test | 0.4496 | 0.4499 |
| large-n-mixed | train | 0.3648 | 0.3647 |
| large-n-sampled | valid | 0.6349 | 0.6336 |
| large-n-sampled | test | 0.4015 | 0.3998 |
| huge-n-mixed | train | 0.5548 | 0.5569 |
| huge-n-sampled | valid | 0.4590 | 0.4593 |
| huge-n-sampled | test | 0.4084 | 0.4084 |

Best Setup by Case
------------------

| Case | Best setup | Evidence |
| --- | --- | --- |
| Overall | current MAP | 0.4585 vs gated REML 0.4608 and raw REML 0.4612 |
| q = 1 | current MAP / gated | 0.3283 vs raw REML 0.3289 |
| q = 2 | current MAP | 0.4531 vs raw/gated REML 0.4534 |
| q >= 3 | current MAP | 0.5534 vs gated REML 0.5585 |
| n >= 2000 | current MAP / gated | 0.4053 vs raw REML 0.4073 |
| n < 2000 | current MAP | beats raw/gated REML in both retained n bins |
| d bins | current MAP | best in all `d <= 4`, `5-8`, and `9+` bins |
| MAP expands MoM/EM | current MAP | 0.4090 vs gated REML 0.4131 |
| MAP shrinks MoM/EM | gated REML, tiny edge | 0.6416 vs current MAP 0.6423 |
| MAP changes MoM/EM by <5% | current MAP | 0.4179 vs gated REML 0.4185 |
| true sigma <0.25 | gated REML, diagnostic only | 0.5967 vs current MAP 0.5988 |

Rejected recompute diagnostics:

| Method | FFX | sRFX | sEps | BLUP |
| --- | ---: | ---: | ---: | ---: |
| current MAP | 0.6696 | 0.4585 | 0.1331 | 0.4978 |
| current MAP + recompute GLS/BLUP | 0.8029 | 0.4585 | 0.1331 | 0.5114 |
| raw-initialized REML + recompute | 0.8017 | 0.4612 | 0.1331 | 0.5118 |
| gated REML + recompute | 0.8016 | 0.4608 | 0.1331 | 0.5115 |
| current-initialized REML + recompute | 0.8052 | 0.4738 | 0.1331 | 0.5102 |

Interpretation
--------------

- Current MAP is the better default. Gated REML has small local wins in a few cells,
  but it does not beat MAP overall or by observable gates.
- Current-initialized REML looked locally promising, improving 11 of 12 cells, but
  medium-n-mixed regressed from 0.5176 to 0.7568 and made the global score worse.
- Optimizing sigma(Eps) inside REML worsened reported sigma(Eps).
- Recomputing GLS/BLUP after refined variance estimates while preserving estimated
  correlations is not viable: it regressed global FFX and BLUP for MAP and every
  REML variant.
- Final decision: keep MAP with diagonal final covariance, retire REML from
  production and package exports, and do not keep a REML diagnostic script in the
  active experiment set.
