Analytical GLMM Experiments
===========================

This directory contains the retained diagnostics and benchmarks for the analytical GLMM
estimator in `metabeta/analytical/`.

Active scripts
--------------

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

- `glmm_inla_comparison.py` — R-INLA reference baseline. Compares the full
  analytical pipeline against R-INLA on Bernoulli or Normal datasets. Supports
  `--analytical-methods raw,current` so normal datasets can compare `lmmNormal`
  against the retained EB-refined normal path and R-INLA.
  Uncorrelated datasets (eta_rfx=0) use independent iid terms with PC priors
  matching HalfNormal(tau_rfx). Correlated datasets (eta_rfx>0, q=2) use the
  iid2d model with a Wishart prior. Reports matched NRMSE and wall time.

Reference files
---------------

- `glmm_perf_baseline.md` — current production benchmark numbers and retired REML
  summary with per-cell comparison tables.
- `estimator_analysis.md` — archived per-stage weakpoint analysis.
- `metabeta/analytical/plan_normal.md` and `plan_bernoulli.md` — current decisions
  and next steps.
