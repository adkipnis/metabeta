Analytical GLMM Experiments
===========================

This directory contains the retained diagnostics and benchmarks for the
analytical GLMM estimator in `metabeta/analytical/`.

- `glmm_error_analysis.py` — verbose per-dataset error diagnostic. Reports NRMSE,
  bias, quantiles, variance-component breakdowns, BLUP calibration, correlation
  quality, and worst cases for a selected dataset split.
- `glmm_raw_diagnostic.py` — required-suite raw MoM/EM attribution diagnostic.
  Compares production MAP and raw MoM/EM with oracle sigma(Eps), beta-for-BLUP,
  Psi substitutions, and MAP-diagonal recomputes to identify which raw stage limits
  accuracy.
- `glmm_required_benchmark.py` — compact required-suite runner. Prints CSV rows for
  FFX, sigma(RFX), sigma(Eps), and BLUP NRMSE on mixed train epochs 1-2 and sampled
  valid/test across small, medium, large, and huge datasets. Use `--sizes` and
  `--methods current raw` for incremental MAP-vs-raw checks.
- `glmm_map_ablation.py` — MAP optimizer ablation diagnostic. Compares four MAP
  variants (sigma_rfx only, rfx+eps, rfx+beta, all three) to confirm that the
  current joint three-parameter optimization is necessary. Result: current is
  Pareto-dominant; simplifying to sigma_rfx-only regresses sRFX by 0.7% with
  nearly no BLUP change; rfx+beta or rfx+eps regress FFX by 2–6%. See
  "Closed: MAP Optimizer Ablation" in `plan.md`.
- `glmm_beta_blend_diagnostic.py` — beta blend sweep diagnostic. Sweeps
  beta_alpha_low (d<=8 gate) and beta_alpha_high (d>8) for both raw and MAP
  paths to confirm that the current OLS blend (0.65/0.75) is optimal. Result:
  every alpha increase degrades BLUP for small/medium; large/huge unaffected.
  See "Closed: Beta Blend" in `plan.md`.
- `glmm_perf_baseline.md` — current benchmark numbers and retired REML summary.
- `estimator_analysis.md` — historical per-stage weakpoint analysis. Covers all
  five stages with open WPs and closed dead ends. Key dead ends: WP-EM3 (EM
  extension), WP-Ψ1 (beta_wg for MoM residuals), WP-EM2 (mom_mask refresh no-op).

Removed historical one-off diagnostics:

- `glmm_shrinkage_diagnostic.py` — I4 shrinkage/oracle BLUP analysis.
- `glmm_beta_leakage_diagnostic.py` — I5 beta-leakage analysis.
- `glmm_alpha_gate_diagnostic.py` — I7 beta-blend alpha schedule sweep.
- `glmm_srfx_diagnostic.py` — sRFX internals replay and floor/cap analysis.
- `glmm_i9_calibration_diagnostic.py` — output-only sigma(RFX) calibration replay.
- `glmm_map_diagnostic.py` — broad MAP sweep superseded by the current production
  benchmark.
- `glmm_reml_diagnostic.py` — REML/profile-MAP variance-scale diagnostic. Retired
  after refreshed data showed current MAP was better globally and recomputing
  GLS/BLUP after refined variances regressed FFX/BLUP.
- `statsmodels_reml.py` — statsmodels REML spot-check. Retired with REML pass.
