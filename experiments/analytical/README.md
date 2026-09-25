Analytical GLMM Experiments
===========================

This directory contains the retained diagnostics and benchmarks for the analytical GLMM
estimator in `metabeta/analytical/`.

Active scripts
--------------

Use `uv run python -u ...` for analytical benchmarks. Several runs
take minutes to hours, and unbuffered output keeps completed dataset blocks visible.

- `glmm_required_benchmark.py` — canonical regression gate. Run before any
  analytical code commit. Prints FFX/sRFX/sEps/BLUP NRMSE CSV for the full
  required suite (mixed-train ×2 + sampled valid/test, all four sizes). Use
  `--methods current raw` for a production-vs-raw comparison, `--methods default` for
  the production path, and `--methods bernoulli_eb` to verify the explicit Bernoulli
  EB preset against the default.

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
- `estimator_analysis.md` — archived per-stage weakpoint analysis.
- `metabeta/analytical/plan_normal.md`, `plan_bernoulli.md`, and `plan_poisson.md` —
  current decisions and next steps.
