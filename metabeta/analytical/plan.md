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

Historical notes in `experiments/analytical/estimator_analysis.md` cover open
weakpoints in all five stages. Key dead ends: WP-EM3 (EM extension, 3 attempts,
all catastrophic on medium-n), WP-Ψ1 (beta_wg for MoM residuals, FFX blowup at
high-d), WP-EM2 (mom_mask refresh is a no-op, G_mom is structurally bounded).

Raw Attribution Diagnostics
----------------------------

Required-suite totals from the oracle attribution pass (`glmm_raw_diagnostic.py`):

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

Remaining Priorities
--------------------

1. [CLOSED] MAP ablation: three-parameter joint optimization confirmed as optimal.
2. [CLOSED] Beta blend: current alpha (d<=8 → 0.65, d>8 → 0.75) confirmed optimal.
3. Monitor only: sigma(Eps) projection and final GLS scale attribution. The retained
   projection change fixed large/huge sigma(Eps) outliers, but the oracle sigma(Eps)
   row does not improve point estimates under the current recompute path.
4. Secondary: EM movement gates or damping. Only pursue after the oracle/stage
   diagnostic shows that EM is the limiting step in broad bins.
5. Secondary: BLUP variance calibration. Keep monitoring, but point-estimate
   accuracy is the current bottleneck.

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

Closed: REML Pass
-----------------

Current MAP: FFX 0.6696, sRFX 0.4585, sEps 0.1331, BLUP 0.4978. Raw/gated REML
worse (sRFX 0.4608–0.4612). Current-initialized REML improved 11/12 cells but
medium-n-mixed regressed from 0.5176 to 0.7568, making global sRFX worse (0.4738).
Recomputing GLS/BLUP after any variant regressed global FFX/BLUP. Decision: keep
MAP, retire REML from package surface. Full cell-level results: `glmm_perf_baseline.md`.

Closed: MAP Optimizer Ablation
--------------------------------

Results from `glmm_map_ablation.py`. Current (all three params) is Pareto-dominant:
FFX 0.6560, sRFX 0.4595, BLUP 0.4733. sigma_rfx-only regresses FFX 0.3%, sRFX 0.7%.
rfx+beta regresses FFX 5.5%; rfx+eps 2.4%. Joint optimization balances variance
attribution; fixing any one parameter forces the others to compensate incorrectly.
Decision: keep three-parameter joint optimization. `map_optimize` kwarg retained.

Closed: Beta Blend
-------------------

Results from `glmm_beta_blend_diagnostic.py`. Every alpha increase from 0.65
monotonically degrades BLUP for small/medium rows; large/huge unaffected (d>8 gate
never triggers for those datasets). Oracle_beta is globally worse (0.5133 vs 0.4978
raw), confirming the current blend is a local optimum. Decision: keep 0.65/0.75.
`beta_alpha_low`/`beta_alpha_high` kwargs retained for diagnostics.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_raw_diagnostic.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
