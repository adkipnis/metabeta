GLMM Analytical Estimator Summary
=================================

Last updated: 2026-05-11 after refreshing current/raw and REML diagnostics.

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
`small|medium|large|huge`. Current production MAP replaces only
`sigma_rfx_est` and the `Psi` diagonal. FFX and BLUP remain raw analytical outputs;
reported sigma(Eps) now comes from the within-group projection estimate, while the
final GLS/BLUP pass is recomputed consistently with that projection scale.

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.1622 | 0.3828 | 0.1658 | 0.4007 |
| small-n-sampled | valid | 0.1983 | 0.4917 | 0.2147 | 0.4771 |
| small-n-sampled | test | 0.1970 | 0.4344 | 0.1973 | 0.4328 |
| medium-n-mixed | train | 0.5092 | 0.5176 | 0.1294 | 0.5465 |
| medium-n-sampled | valid | 0.3938 | 0.4144 | 0.1853 | 0.4997 |
| medium-n-sampled | test | 0.4954 | 0.4496 | 0.1755 | 0.5042 |
| large-n-mixed | train | 0.9218 | 0.3648 | 0.1117 | 0.5187 |
| large-n-sampled | valid | 0.9254 | 0.6349 | 0.1344 | 0.5603 |
| large-n-sampled | test | 0.7178 | 0.4015 | 0.1329 | 0.5122 |
| huge-n-mixed | train | 0.9782 | 0.5548 | 0.0942 | 0.4890 |
| huge-n-sampled | valid | 1.1043 | 0.4590 | 0.1197 | 0.5214 |
| huge-n-sampled | test | 1.3469 | 0.4084 | 0.1171 | 0.5073 |

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
- The refreshed raw baseline has identical FFX, sigma(Eps), and BLUP to current
  MAP by construction, while current MAP improves sigma(RFX) in every required
  cell. Raw MoM/EM is therefore not a competitive global sigma(RFX) fallback.

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

Recompute diagnostic:

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
- Recomputing GLS/BLUP after refined variance estimates is not viable: it regressed
  global FFX and BLUP for MAP and every REML variant.
- Final decision: keep output-local MAP, retire REML from production and package
  exports, and do not keep a REML diagnostic script in the active experiment set.
