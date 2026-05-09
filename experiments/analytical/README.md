Analytical GLMM Experiments
===========================

This directory contains diagnostics and compact benchmarks for the analytical
GLMM estimator in `metabeta/analytical/`.

- `glmm_error_analysis.py` — verbose per-dataset error diagnostic. Reports NRMSE,
  bias, quantiles, variance-component breakdowns, BLUP calibration, correlation
  quality, and worst cases for a selected dataset split.
- `glmm_required_benchmark.py` — compact required-suite runner. Prints CSV rows for
  FFX, sigma(RFX), sigma(Eps), and BLUP NRMSE on mixed train epochs 1-2 and sampled
  valid/test across small, medium, large, and huge datasets.
- `glmm_shrinkage_diagnostic.py` — I4 diagnostic. Compares estimated vs true BLUP
  shrinkage for q=1 rows and runs oracle BLUP ablations separating beta, Psi, and
  sigma_eps effects.
- `glmm_beta_leakage_diagnostic.py` — I5 diagnostic. Measures fixed-effect leakage
  into BLUP residuals, bins BLUP errors by beta/projection/rank diagnostics, and runs
  beta-OLS blend ablations.
- `glmm_perf_baseline.md` — running fix log and benchmark record for analytical GLMM
  estimator changes.
