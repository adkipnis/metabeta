Analytical GLMM Plan
====================

Last updated: 2026-05-11.

Current Decision
----------------

The next optimization pass should start with the raw analytical estimator:
Gaussian MoM/EM, GLS, BLUP, and uncertainty outputs with `map_refine=False`.
MAP and gated REML remain retained candidates, but they should be evaluated only
after the raw estimator failure modes are measured and, where possible, fixed.

This sequencing matters because production MAP currently replaces only
`sigma_rfx_est` and the `Psi` diagonal. FFX, sigma(Eps), BLUP, and BLUP variance
are still inherited from the raw MoM/EM pipeline, so those outputs cannot improve
until the raw stages improve.

Stable Baseline
---------------

- Current production baseline: `glmm()` with the MAP sigma(RFX) refinement.
- Raw baseline for the next work cycle: `glmm(..., map_refine=False)`.
- Retained post-raw candidate: gated REML/profile-MAP over sigma(RFX), initialized
  from raw MoM/EM.
- Retained context-only candidate: Laplace curvature around the MAP/REML optimum.

First raw-estimator pass retained two changes:

- report and use the within-group projection sigma(Eps) in the final GLS/BLUP pass;
- reduce the low-dimensional final BLUP residual beta blend from 100% pooled OLS
  to 65% pooled OLS.

The high-dimensional beta blend was tested at 0.65 and 0.50 and rejected because
large-valid or huge-mixed BLUPs regressed. The high-dimensional branch remains
0.75.

Historical notes in `estimator_analysis.md` point to four raw-estimator areas that
still deserve direct measurement:

- Stage 1 sigma(Eps): within-group projection df and rank handling.
- Stage 2 Psi: noisy low-information MoM diagonal/off-diagonal estimates.
- Stage 3 GLS/BLUP: fixed-effect leakage into BLUP residuals and beta fallback
  behavior.
- Stage 4/5 EM and BLUP variance: biased EM fixed points, sigma(Eps) drift, and
  residual under-calibration of BLUP uncertainty.

Diagnostic Plan: Raw Estimators First
-------------------------------------

Add or extend an experiment-only raw diagnostic, tentatively
`experiments/analytical/glmm_raw_diagnostic.py`. It should not change
`metabeta/analytical/glmm.py` while diagnosing. The diagnostic should run the
required Gaussian suite and collect row-level/stage-level data for the raw path.

Required estimator rows:

- `raw`: `glmm(..., map_refine=False)`.
- `current`: production `glmm()` with MAP enabled, reported only as a reference.
- `oracle_sigma_eps`: raw path with true sigma(Eps) substituted where practical.
- `oracle_beta_blup`: raw path with true beta used only for final BLUP residuals.
- `oracle_psi`: raw path with true Psi used for GLS/BLUP, to quantify the remaining
  variance-component ceiling.

The oracle rows are diagnostic only. They are intended to identify which raw stage
limits accuracy, not to define production behavior.

Required metrics:

- Standard NRMSE for FFX, sigma(RFX), sigma(Eps), and BLUP.
- Bias, absolute-error quantiles, and signed error for sigma(Eps) and sigma(RFX).
- BLUP variance calibration by group-size bins.
- Runtime per row for raw MoM/EM and production MAP reference.
- Failure/fallback rates: finite checks, active masks, rank-deficient groups,
  low-information MoM rows, EM early exit, cap/clamp rates, and beta fallback rates.

Required breakdowns:

- Shape: `d`, `q`, `m`, `n`, median/min/max `n_i`, and `n / m`.
- Identification: `G_mom`, `G_mom - d`, `enough_full_mom`, componentwise Psi counts,
  summed `z_rank`, `mx_rank`, and residual df.
- Signal: true sigma(Eps), mean true sigma(RFX), R2/SNR proxy, eta/correlation mode,
  and MAP-vs-raw sigma(RFX) direction for reference.
- Error interactions: beta projection error vs BLUP error, sigma(Eps) error vs
  sigma(RFX) error, and Psi diagonal underestimation vs BLUP shrinkage.

Stage-Specific Questions
------------------------

1. sigma(Eps) projection:
   - Does the within-group estimator become biased when residual df is small or
     when `mx_rank` is far below active `d`?
   - Are row-level sigma(Eps) errors mainly structural (`n_i`, rank, df) or signal
     driven (R2/SNR, true sigma(Eps))?
   - Does anchoring EM closer to the projection estimate help or hurt the rows
     where sigma(Eps) currently regresses?

2. Initial Psi MoM:
   - Are sigma(RFX) errors dominated by rows with low `G_mom`, low component counts,
     high `q`, or high correlation?
   - Does the componentwise diagonal path help high-q rows, or does it introduce a
     distinct bias relative to joint MoM?
   - Are off-diagonal estimates adding useful information for BLUPs, or mostly noise?

3. GLS beta and BLUP:
   - Which beta candidate is best by case: pooled OLS, within-group beta, GLS beta,
     or the current beta-for-BLUP blend?
   - Is the current beta blend still appropriate after the regenerated simulation
     data and updated prior coverage?
   - Are high BLUP errors mostly caused by beta leakage, Psi shrinkage, or both?

4. EM refinement:
   - Which rows move toward better Psi/sigma(Eps) after each EM iteration, and which
     rows move toward a biased fixed point?
   - Do trim/cap rules activate in the same rows where they help?
   - Is there a measurable gate that can stop EM or damp EM without repeating past
     failed broad iteration-count changes?

5. BLUP variance:
   - Does the current delta/KH/floor stack remain calibrated after the latest data
     regeneration?
   - Which bins remain under-calibrated: low `G_mom`, large `n_i`, high `q`, or
     underestimated Psi?

Acceptance Criteria for Raw Changes
-----------------------------------

A raw-estimator change should be considered only if it improves at least one
primary output on the required suite without material regressions elsewhere:

- FFX, sigma(Eps), and BLUP improvements are more important than sigma(RFX) alone,
  because MAP/REML can refine only the sigma(RFX) report afterward.
- A sigma(RFX) raw improvement is still valuable if it improves BLUP or reduces the
  number of rows where MAP/REML are needed.
- Any candidate must be compared against both `raw` and `current` production MAP.
- Changes that only improve oracle-like rows or a single narrow bin stay as
  experiments unless they define a clear gate.

Second Pass: MAP/REML After Raw
-------------------------------

After raw-estimator diagnostics are complete:

- Re-run the retained gated REML diagnostic against the updated raw baseline.
- Re-check the three-way question: raw MoM/EM vs current MAP vs gated REML.
- Keep REML output-local unless recomputing GLS/BLUP after REML preserves FFX,
  sigma(Eps), and BLUP.
- Consider a no-refinement branch only if the raw diagnostic confirms stable,
  observable cases where both MAP and REML are worse than raw MoM/EM.
- Keep Laplace curvature as a context/uncertainty feature candidate rather than a
  point-estimate improvement path.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run python experiments/analytical/glmm_reml_diagnostic.py --breakdown
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
