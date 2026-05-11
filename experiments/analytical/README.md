Analytical GLMM Experiments
===========================

This directory contains the retained diagnostics and benchmarks for the
analytical GLMM estimator in `metabeta/analytical/`.

- `glmm_error_analysis.py` — verbose per-dataset error diagnostic. Reports NRMSE,
  bias, quantiles, variance-component breakdowns, BLUP calibration, correlation
  quality, and worst cases for a selected dataset split.
- `glmm_required_benchmark.py` — compact required-suite runner. Prints CSV rows for
  FFX, sigma(RFX), sigma(Eps), and BLUP NRMSE on mixed train epochs 1-2 and sampled
  valid/test across small, medium, large, and huge datasets. Use `--sizes` and
  `--methods current raw` for incremental MAP-vs-raw checks.
- `glmm_reml_diagnostic.py` — experiment-only REML/profile-MAP variance-scale
  diagnostic. It compares current production MAP, MoM/EM, and the retained gated
  REML-vs-MAP candidate. Use `--breakdown` for binned raw/gated REML deltas,
  gate rates, fallback/clamp rates, and MoM/EM comparison.
- `statsmodels_reml.py` — spot-checks analytical GLMM estimates against
  statsmodels REML on individual datasets.
- `glmm_perf_baseline.md` — concise current benchmark and gated REML summary.

Removed historical one-off diagnostics:

- `glmm_shrinkage_diagnostic.py` — I4 shrinkage/oracle BLUP analysis.
- `glmm_beta_leakage_diagnostic.py` — I5 beta-leakage analysis.
- `glmm_alpha_gate_diagnostic.py` — I7 beta-blend alpha schedule sweep.
- `glmm_srfx_diagnostic.py` — sRFX internals replay and floor/cap analysis.
- `glmm_i9_calibration_diagnostic.py` — output-only sigma(RFX) calibration replay.
- `glmm_map_diagnostic.py` — broad MAP sweep superseded by the retained gated
  REML diagnostic and current production benchmark.
