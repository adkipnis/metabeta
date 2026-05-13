Analytical GLMM Experiments
===========================

This directory contains the retained diagnostics and benchmarks for the
analytical GLMM estimator in `metabeta/analytical/`.

Active scripts
--------------

- `glmm_required_benchmark.py` — canonical regression gate. Run before any
  analytical code commit. Prints FFX/sRFX/sEps/BLUP NRMSE CSV for the full
  required suite (mixed-train ×2 + sampled valid/test, all four sizes). Use
  `--methods current raw` for a MAP-vs-raw comparison.

- `glmm_error_analysis.py` — the only calibration diagnostic. Run this when
  verifying `blup_var` or checking interval coverage. Reports BLUP coverage ratios
  (mean err² / mean blup_var) by group-size bin, alongside NRMSE, bias, quantiles,
  variance-component breakdowns, and worst-case examples for a single dataset.

- `glmm_raw_diagnostic.py` — oracle attribution reference. Run this when
  investigating whether a raw-stage change would close the BLUP gap. Answers
  "which stage limits accuracy?" via oracle sigma(Eps), beta, and Psi substitutions.

Reference files
---------------

- `glmm_perf_baseline.md` — current production benchmark numbers and retired REML
  summary with per-cell comparison tables.
- `estimator_analysis.md` — historical per-stage weakpoint analysis. Key dead ends:
  WP-EM3 (EM extension, 3 tries all catastrophic), WP-Ψ1 (beta_wg for MoM
  residuals blows up at high-d), WP-EM2 (mom_mask refresh is a structural no-op).

Removed diagnostics
-------------------

Scripts removed after single use; results and decisions are documented in
`metabeta/analytical/plan_normal.md` and `plan_bernoulli.md`.

- `glmm_map_ablation.py` — MAP optimizer ablation. Confirmed three-parameter joint
  optimization is Pareto-dominant over all subsets.
- `glmm_beta_blend_diagnostic.py` — beta blend sweep. Confirmed 0.65/0.75 alpha
  gate is optimal; oracle_beta globally worse than current.

Scripts removed earlier after their diagnostic pass was complete.

- `glmm_shrinkage_diagnostic.py` — I4 shrinkage/oracle BLUP analysis.
- `glmm_beta_leakage_diagnostic.py` — I5 beta-leakage analysis.
- `glmm_alpha_gate_diagnostic.py` — I7 beta-blend alpha schedule sweep.
- `glmm_srfx_diagnostic.py` — sRFX internals replay and floor/cap analysis.
- `glmm_i9_calibration_diagnostic.py` — output-only sigma(RFX) calibration replay.
- `glmm_map_diagnostic.py` — broad MAP sweep superseded by the required benchmark.
- `glmm_reml_diagnostic.py` — REML/profile-MAP variance-scale diagnostic.
- `statsmodels_reml.py` — statsmodels REML spot-check. Retired with REML pass.
