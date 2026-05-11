Analytical GLMM Plan
====================

Last updated: 2026-05-11.

Current Decision
----------------

The current production baseline is `glmm()` with MAP sigma(RFX) refinement. The
raw MoM/EM pass retained two low-risk improvements, and the subsequent REML pass
did not find a production-worthy replacement or gate. REML support has therefore
been retired from the analytical package surface.

The key decision is output-local MAP only: MAP replaces `sigma_rfx_est` and the
`Psi` diagonal. Recomputing final GLS/BLUP after MAP or REML was tested and
rejected because it regressed global FFX and BLUP.

Stable Baseline
---------------

- Current production baseline: `glmm()` with the MAP sigma(RFX) refinement.
- Raw baseline for the next work cycle: `glmm(..., map_refine=False)`.
- Retired production candidate: gated REML/profile-MAP over sigma(RFX).
- Retained context-only candidate: Laplace curvature around the MAP optimum.

First raw-estimator pass retained two changes:

- report and use the within-group projection sigma(Eps) in the final GLS/BLUP pass;
- reduce the low-dimensional final BLUP residual beta blend from 100% pooled OLS
  to 65% pooled OLS.

The high-dimensional beta blend was tested at 0.65 and 0.50 and rejected because
large-valid or huge-mixed BLUPs regressed. The high-dimensional branch remains
0.75.

Historical notes in `estimator_analysis.md` point to four raw-estimator areas that
can still be revisited later:

- Stage 1 sigma(Eps): within-group projection df and rank handling.
- Stage 2 Psi: noisy low-information MoM diagonal/off-diagonal estimates.
- Stage 3 GLS/BLUP: fixed-effect leakage into BLUP residuals and beta fallback
  behavior.
- Stage 4/5 EM and BLUP variance: biased EM fixed points, sigma(Eps) drift, and
  residual under-calibration of BLUP uncertainty.

Deferred Raw Diagnostics
------------------------

If MAP stalls, add or extend an experiment-only raw diagnostic, tentatively
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
  because output-local MAP can refine only the sigma(RFX) report afterward.
- A sigma(RFX) raw improvement is still valuable if it improves BLUP or reduces the
  number of rows where MAP is needed.
- Any candidate must be compared against both `raw` and `current` production MAP.
- Changes that only improve oracle-like rows or a single narrow bin stay as
  experiments unless they define a clear gate.

Closed REML/MAP Pass
--------------------

The MAP/REML pass is complete. Current MAP remains the production baseline.

Required-suite results:

- Current MAP: FFX 0.6696, sRFX 0.4585, sEps 0.1331, BLUP 0.4978.
- Raw MoM/EM: same FFX/sEps/BLUP as current MAP, but sRFX 0.7103.
- Raw-initialized REML: sRFX 0.4612, worse than current MAP.
- Gated REML: sRFX 0.4608, worse than current MAP.
- REML with sigma(Eps) optimized: sRFX 0.4608 and sEps regressed to 0.1360.
- Current-initialized REML: improved 11 of 12 required cells, but medium-n-mixed
  regressed from 0.5176 to 0.7568, making global sRFX worse at 0.4738.

Recompute diagnostics were also rejected:

- Recomputing GLS/BLUP after current MAP kept sRFX/sEps unchanged but regressed
  global FFX from 0.6696 to 0.8029 and BLUP from 0.4978 to 0.5114.
- Recomputing after REML variants likewise regressed global FFX/BLUP.

Decision:

- Keep current output-local MAP.
- Do not integrate REML.
- Remove REML from the package surface and retire the REML diagnostic script.
- Keep Laplace curvature as a later context/uncertainty feature candidate around
  the MAP optimum, not as a point-estimate path.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
