Analytical GLMM Experiments
===========================

This directory contains the retained diagnostics and benchmarks for the
analytical GLMM estimator in `metabeta/analytical/`.

- `glmm_error_analysis.py` — verbose per-dataset error diagnostic. Reports NRMSE,
  bias, quantiles, variance-component breakdowns, BLUP calibration, correlation
  quality, and worst cases for a selected dataset split.
- `glmm_required_benchmark.py` — compact required-suite runner. Prints CSV rows for
  FFX, sigma(RFX), sigma(Eps), and BLUP NRMSE on mixed train epochs 1-2 and sampled
  valid/test across small, medium, large, and huge datasets.
- `glmm_map_diagnostic.py` — experiment-only comparison of MoM/EM baseline
  against hybrid MAP global refinements that keep MoM/EM sigma(Eps) and BLUP
  outputs. It reports FFX-only, sRFX-only, and FFX+sRFX hybrids; optional
  full-likelihood MAP rows are available for completeness.
- `glmm_perf_baseline.md` — concise final fix summary: accepted changes, rejected
  experiments, final benchmark, and remaining weaknesses.

Removed historical one-off diagnostics:

- `glmm_shrinkage_diagnostic.py` — I4 shrinkage/oracle BLUP analysis.
- `glmm_beta_leakage_diagnostic.py` — I5 beta-leakage analysis.
- `glmm_alpha_gate_diagnostic.py` — I7 beta-blend alpha schedule sweep.
- `glmm_srfx_diagnostic.py` — sRFX internals replay and floor/cap analysis.
- `glmm_i9_calibration_diagnostic.py` — output-only sigma(RFX) calibration replay.
