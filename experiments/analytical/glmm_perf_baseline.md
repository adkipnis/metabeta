GLMM Analytical Estimator Summary
=================================

Last updated: 2026-05-17 after promoting normal EB to the retained Gaussian path.

This file records the current retained Gaussian GLMM analytical benchmark and the
closed REML diagnostic decision. Historical MAP/REML sweep scripts were removed;
the reusable scripts are documented in `README.md`.

Current Production EB Benchmark
-------------------------------

Command:

```bash
uv run python experiments/analytical/glmm_required_benchmark.py --methods current raw
```

Required suite: mixed train epochs 1-2 and sampled valid/test for
`small|medium|large|huge`. The current production normal path uses the MAP stage
internally, reports prior-capped medium+ β, then applies the one-shot diagonal EB
σ_rfx update before final BLUP output. Reported sigma(Eps) still comes from the
within-group projection estimate. The existing calibrated `blup_var` output is
preserved pending a separate variance-calibration pass.

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

Normal Laplace-EB Prototype
---------------------------

R-INLA is being used as a slow reference for the normal path, but not as a backend.
The exact correlated Gaussian INLA specification has shown numerical failures on
some datasets, so the comparison uses a diagonal random-effects INLA reference to
match the production final covariance assumption.

The retained normal path now carries the optimized MAP β for `d > 4`; `d <= 4` keeps the
older GLS/OLS final β because direct MAP β overfits those small rows. The reported
medium+ β is prior-capped at `ν_ffx ± 4τ_ffx`, while BLUP residuals continue to use the
uncapped MAP β to preserve BLUP accuracy. The variance-scale add-on is a normal-specific
diagonal Laplace-EB calibration, now the default normal GLMM answer and benchmarked as
`normal_eb`:

- use the exact Gaussian marginal likelihood rather than a Bernoulli-style nested
  random-effect optimizer;
- update only diagonal `sigma_rfx` with a posterior-moment EB update plus prior
  pseudo-count;
- keep the EB σ update one-shot; β comes from the existing MAP pass for `d > 4`, with a
  prior-aware reporting cap for rare FFX tail failures;
- accept only objective-improving, finite updates, otherwise return current MAP;
- promote only if the required-suite BLUP improves without FFX/σ(Eps) regressions.

First-1000 required-suite result after the β cap patch: the internal MAP β stage improves
FFX sharply on every medium/large/huge row, and EB improves σ(RFX) and BLUP in all 12 rows
while keeping σ(Eps) unchanged. Runtime remains single-digit milliseconds per dataset.
Detailed row table is in `metabeta/analytical/plan_normal.md`.

Retired REML Diagnostics
------------------------

REML/profile-MAP was tested as an experiment-only variance-scale refinement and
retired after the refreshed benchmark. The diagnostic script and package-level
REML support were removed; MAP is retained only as the internal initializer/refinement
stage for the EB path.

Full-suite row-weighted sRFX NRMSE:

| Method | sRFX |
| --- | ---: |
| mom_em | 0.7103 |
| current MAP | 0.4585 |
| raw-initialized REML | 0.4612 |
| gated REML | 0.4608 |
| REML with sigma(Eps) optimized | 0.4608 |
| current-initialized REML | 0.4738 |

Required-suite cells (current MAP vs gated REML):

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

Decision: keep current MAP. Gated REML has negligible local wins in 3 cells but is
globally worse. Current-initialized REML improved 11/12 cells but regressed
medium-n-mixed from 0.5176 to 0.7568. Recomputing GLS/BLUP after refined variances
is not viable: regressed global FFX from 0.6696 to 0.8029 and BLUP from 0.4978 to
0.5114 across all variants.
