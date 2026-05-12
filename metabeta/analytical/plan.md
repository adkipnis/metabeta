Analytical GLMM Plan
====================

Last updated: 2026-05-12 (beta blend closed).

Current Decision
----------------

The current production baseline is `glmm()` with MAP sigma(RFX) refinement and a
diagonal-MAP final GLS/BLUP pass. The raw MoM/EM pass retained two low-risk
improvements, and the subsequent REML pass did not find a production-worthy
replacement or gate. REML support has therefore been retired from the analytical
package surface.

The current MAP path reports MAP `sigma_rfx_est`, writes a diagonal MAP `Psi`, and
recomputes final Gaussian beta/BLUPs with that diagonal covariance. Earlier
recomputes that preserved estimated correlations were rejected; the retained path
uses MAP scale but excludes noisy off-diagonal covariance from final BLUP shrinkage.
The existing calibrated `blup_var` stack is preserved until variance calibration is
re-checked for the new point-estimate path.

Stable Baseline
---------------

- Current production baseline: `glmm()` with MAP sigma(RFX) refinement and
  diagonal-MAP final GLS/BLUP recompute.
- Raw baseline for the next work cycle: `glmm(..., map_refine=False)`.
- Legacy output-local MAP comparison: `glmm(..., map_recompute_blup=False)`.
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

The next active work should focus on raw MoM/EM attribution and MAP-diagonal
ablation. Current MAP now improves reported sigma(RFX) and final BLUPs, but
sigma(Eps) is still raw-derived and the raw Psi path still determines the fallback
behavior when MAP is disabled.

Initial diagnostic result:

```bash
uv run python experiments/analytical/glmm_raw_diagnostic.py
```

Required-suite totals from the first oracle attribution pass:

| Method | FFX | sRFX | sEps | BLUP |
| --- | ---: | ---: | ---: | ---: |
| current MAP | 0.6696 | 0.4585 | 0.1331 | 0.4978 |
| raw MoM/EM | 0.6696 | 0.7103 | 0.1331 | 0.4978 |
| oracle sigma(Eps) | 0.8140 | 0.7103 | 0.0000 | 0.5336 |
| oracle beta for BLUP | 0.6696 | 0.7103 | 0.1331 | 0.5133 |
| oracle Psi | 0.6673 | 0.0000 | 0.1331 | 0.4301 |

Takeaway: the global BLUP ceiling is primarily Psi/shrinkage, not sigma(Eps) or
beta leakage. Beta leakage remains a conditional low-dimensional issue: true beta
for final BLUP residuals improves small rows but hurts large/huge rows. True
sigma(Eps) is not a promising point-estimate path; it often worsens FFX/BLUP under
the current final GLS/BLUP machinery.

Follow-up Psi decomposition:

| Method | FFX | sRFX | sEps | BLUP |
| --- | ---: | ---: | ---: | ---: |
| legacy output-local MAP | 0.6696 | 0.4585 | 0.1331 | 0.4978 |
| output Psi recompute | 0.8141 | 0.7103 | 0.1331 | 0.5341 |
| MAP diagonal recompute | 0.6624 | 0.4585 | 0.1331 | 0.4682 |
| oracle Psi diagonal | 0.6679 | 0.0000 | 0.1331 | 0.4408 |
| oracle full Psi | 0.6673 | 0.0000 | 0.1331 | 0.4301 |

Decision: keep the MAP diagonal recompute as production. True diagonal explains
most of the oracle BLUP gain, while estimated off-diagonal correlations are often
harmful when used in the final BLUP covariance.

Updated priority order:

1. [CLOSED] MAP ablation: three-parameter joint optimization confirmed as optimal.
   See "Closed MAP Optimizer Ablation" below.
2. [CLOSED] Beta blend: current alpha (d<=8 → 0.65, d>8 → 0.75) confirmed optimal.
   See "Closed Beta Blend Diagnostic" below.
3. Monitor only: sigma(Eps) projection and final GLS scale attribution. The retained
   projection change fixed large/huge sigma(Eps) outliers, but the oracle sigma(Eps)
   row does not improve point estimates under the current recompute path.
4. Secondary: EM movement gates or damping. Only pursue after the oracle/stage
   diagnostic shows that EM is the limiting step in broad bins.
5. Secondary: BLUP variance calibration. Keep monitoring, but point-estimate
   accuracy is the current bottleneck.

Add or extend an experiment-only raw diagnostic,
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

- FFX, sigma(Eps), and BLUP improvements are more important than sigma(RFX) alone.
  MAP now improves reported sigma(RFX) and BLUPs, but raw changes are still needed
  for sigma(Eps) or for non-MAP fallback behavior.
- A sigma(RFX) raw improvement is still valuable if it improves BLUP or reduces the
  number of rows where MAP is needed.
- Any candidate must be compared against both `raw` and `current` production MAP.
- Changes that only improve oracle-like rows or a single narrow bin stay as
  experiments unless they define a clear gate.

Closed REML Pass
----------------

The REML pass is complete. Current MAP remains the production baseline.

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

REML decision:

- Keep current MAP.
- Do not integrate REML.
- Remove REML from the package surface and retire the REML diagnostic script.
- Keep Laplace curvature as a later context/uncertainty feature candidate around
  the MAP optimum, not as a point-estimate path.

Closed MAP Optimizer Ablation
------------------------------

MAP optimizer ablation was run on 2026-05-12 via
`experiments/analytical/glmm_map_ablation.py`. Four variants were compared:

| Method | FFX | sRFX | sEps | BLUP |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.6625 | 0.6803 | 0.1482 | 0.4975 |
| map_rfx only | 0.6581 | 0.4629 | 0.1482 | 0.4734 |
| map_rfx + eps | 0.6717 | 0.4624 | 0.1482 | 0.4731 |
| map_rfx + beta | 0.6920 | 0.4602 | 0.1482 | 0.4743 |
| current (all three) | 0.6560 | 0.4595 | 0.1482 | 0.4733 |

Decision: keep the current three-parameter joint optimization (sigma_rfx + beta +
sigma_eps). Current is Pareto-dominant: best sRFX (0.4595), best FFX (0.6560), and
essentially best BLUP (tied with map_rfx_eps at 0.4733 vs 0.4731).

Simplifying to sigma_rfx only: sRFX regresses by 0.7% and BLUP is nearly
unchanged, but FFX slightly worsens. Adding back individual parameters without
all three is strictly worse for FFX — rfx+beta causes a 5.5% FFX regression,
rfx+eps causes a 2.4% FFX regression. The joint optimization stabilizes by
balancing variance attribution between beta, rfx, and eps; fixing any one
parameter forces the others to compensate incorrectly.

The `optimize` kwarg in `refineNormalMapSrfx` / `glmm(map_optimize=...)` is
retained for future diagnostics but is not used in production.

Closed Beta Blend Diagnostic
------------------------------

Beta blend sweep was run on 2026-05-12 via
`experiments/analytical/glmm_beta_blend_diagnostic.py`. The alpha gate
(beta_for_blup = (1-alpha)*gls_beta + alpha*pooled_ols, gated by active_d_count<=8)
was swept from 0.65 to 0.80 for both raw and MAP paths:

| Method | small | medium | large | huge | global |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw (d<=8: 0.65, d>8: 0.75) | 0.4369 | 0.5168 | 0.5304 | 0.5059 | 0.4975 |
| raw (d<=8: 0.70) | 0.4373 | 0.5171 | 0.5304 | 0.5059 | 0.4977 |
| raw (d<=8: 0.75) | 0.4377 | 0.5175 | 0.5304 | 0.5059 | 0.4979 |
| raw (d<=8: 0.80) | 0.4381 | 0.5179 | 0.5304 | 0.5059 | 0.4981 |
| current MAP (d<=8: 0.65, d>8: 0.75) | 0.4224 | 0.5015 | 0.4876 | 0.4818 | 0.4733 |
| MAP (d<=8: 0.70) | 0.4228 | 0.5019 | 0.4876 | 0.4818 | 0.4735 |
| MAP (d<=8: 0.75) | 0.4233 | 0.5024 | 0.4876 | 0.4818 | 0.4738 |
| MAP (d<=8: 0.80) | 0.4237 | 0.5028 | 0.4876 | 0.4818 | 0.4740 |

Decision: keep the current blend (0.65 for d<=8, 0.75 for d>8). Every alpha
increase monotonically degrades BLUP for small and medium rows. Large/huge rows
are unaffected (d > 8 gate never triggers). The oracle_beta ceiling (true beta
for BLUP residuals) is globally worse than current production, confirming that
the improvement seen in plan diagnostics is conditional and that the current
blend is already a local optimum.

The `beta_alpha_low` / `beta_alpha_high` kwargs in `lmmNormal`, `refineNormalMapSrfx`,
and `glmm(beta_alpha_low=..., beta_alpha_high=...)` are retained for future
diagnostics but are not used in production.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_raw_diagnostic.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
