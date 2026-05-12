Analytical GLMM Plan
====================

Last updated: 2026-05-12 (BLUP calibration active).

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
5. [CLOSED] BLUP variance calibration. See below.

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

Closed: BLUP Variance Calibration
-----------------------------------

Structural mismatch: `_recomputeNormalFinalDiagMap` used MAP Psi for BLUP point
estimates but left `blup_var` from the raw stats dict unchanged. Raw blup_var was
dominated by the `Ψ/G_mom` additive floor (calibrated for raw underestimated Psi);
under MAP (larger Psi), that floor overcorrected, producing calibration ratios of
0.23–0.86 across n_g bins (underconfident — intervals too wide by 1.2–4.4×).

Fix (map.py `_recomputeNormalFinalDiagMap`): recompute blup_var from MAP W_g
(= σ²·MAP_W_g diagonal from the existing `_normalGlsAndBlups` call) with
delta-method + KH corrections, floor at Ψ_diag/(2·n_g). Dropped the additive
Ψ/G_mom floor — under MAP, Psi is better estimated and the floor was overcorrecting.
Added `ns` to `refineNormalMapSrfx` signature and forwarded from `glmm.py`.

Post-fix calibration ratios (mean(err²)/mean(blup_var) by n_g bin):

| n_g bin | small | large | huge |
| --- | ---: | ---: | ---: |
| 5–9 | 0.91 | 1.09 | 1.03 |
| 9–13 | 0.89 | 0.98 | 0.94 |
| 13–17 | 0.96 | 0.93 | 0.85 |
| 17–25 | 0.97 | 0.93 | 1.36 |
| 25–150 | 1.08 | 1.07 | 1.13 |

12/15 bins within ±10% of 1.0. Worst remaining: huge n_g 17-24 at 1.36 (previously
0.79 before, slightly overconfident now). All 12 point-estimate benchmark cells
unchanged. Before fix, worst bins were 0.23–0.26 (intervals 4× too wide).

Closed: REML Pass
-----------------

Current MAP: FFX 0.6696, sRFX 0.4585, sEps 0.1331, BLUP 0.4978. Raw/gated REML
worse (sRFX 0.4608–0.4612). Current-initialized REML improved 11/12 cells but
medium-n-mixed regressed from 0.5176 to 0.7568, making global sRFX worse (0.4738).
Recomputing GLS/BLUP after any variant regressed global FFX/BLUP. Decision: keep
MAP, retire REML from package surface. Full cell-level results: `glmm_perf_baseline.md`.

Closed: MAP Optimizer Ablation
--------------------------------

Results from `glmm_map_ablation.py` (script retired). Current (all three params) is
Pareto-dominant: FFX 0.6560, sRFX 0.4595, BLUP 0.4733. sigma_rfx-only regresses
FFX 0.3%, sRFX 0.7%. rfx+beta regresses FFX 5.5%; rfx+eps 2.4%. Joint optimization
balances variance attribution; fixing any one parameter forces the others to
compensate incorrectly. Decision: keep three-parameter joint optimization.
`map_optimize` kwarg retained for future diagnostics.

Closed: Beta Blend
-------------------

Results from `glmm_beta_blend_diagnostic.py` (script retired). Every alpha increase
from 0.65 monotonically degrades BLUP for small/medium rows; large/huge unaffected
(d>8 gate never triggers for those datasets). Oracle_beta is globally worse (0.5133
vs 0.4978 raw), confirming the current blend is a local optimum. Decision: keep
0.65/0.75. `beta_alpha_low`/`beta_alpha_high` kwargs retained for future diagnostics.

Bernoulli PQL Baseline
----------------------

Last measured: 2026-05-12. Estimator: `lmmBernoulli` (6 PQL passes, unregularized
pooled IRLS β₀, no MAP refinement). Metric: NRMSE (same as NN eval). sEps is n/a
(Bernoulli has no residual variance). Collected via `glmm_error_analysis.py` over
2 mixed train epochs + sampled valid for each size.

| Dataset | Partition | FFX | sRFX | BLUP |
| --- | --- | ---: | ---: | ---: |
| small-b-mixed | train | 0.7898 | 0.6275 | 0.6384 |
| small-b-sampled | valid | 1.1875 | 0.6873 | 0.6828 |
| medium-b-mixed | train | 1.5274 | 0.7236 | 0.7667 |
| medium-b-sampled | valid | 1.6387 | 0.7652 | 0.7901 |
| large-b-mixed | train | 2.0386 | 0.8465 | 0.8809 |
| large-b-sampled | valid | 2.8019 | 0.9498 | 1.0738 |
| huge-b-mixed | train | 3.1533 | 0.9895 | 1.2625 |
| huge-b-sampled | valid | 3.4570 | 1.0995 | 1.4105 |

Root cause analysis from `glmm_error_analysis.py` (all sizes, train + valid):

**FFX (β) — dominant failure mode.** NRMSE scales from 0.79 (small-train) to 3.46
(huge-valid). Root cause: Bernoulli binary outcomes carry less Fisher information per
observation than Normal (working weight W = μ(1−μ) ≤ 0.25 vs 1/σ²). At large d
(9–16), the pooled IRLS is severely underdetermined at low n/d. FFX breakdown by n/d
at huge-b-sampled: RMSE 5.08 at n/d < 19 (lowest quartile). The GLS refinement
cannot recover because there is no residual σ² lever — all variation is in the binary
outcome.

**σ_rfx — systematic bidirectional bias.** Consistent across all sizes:
- Low true σ_rfx (≤ 0.25): upward bias +0.30–0.36 (Laplace psi_0 floor at 0.25
  prevents collapse but overshoots small true values)
- High true σ_rfx (≥ 0.85): downward bias −0.23–0.46 (Laplace M-step shrinks toward
  the group average; underestimates large variance components)
- NRMSE 0.63–1.10; 0% clip rate (Laplace never collapses to zero unlike MoM)

**BLUPs — tracks FFX.** NRMSE 0.64 (small-train) to 1.41 (huge-valid). Small-group
(n_g < 10) BLUP RMSE is 0.67–1.05 across sizes. BLUP quality is gated by FFX
leakage: bad β contaminates the BLUP residual (ỹ − Xβ), pulling BLUPs away from
the true random effects.

**blup_var calibration.** Small-b-mixed calibration ratios (mean err²/mean blup_var):
0.77–1.31 by n_g bin (n_g 5–150). Small groups slightly overconfident (1.31 at
n_g=5-9), large groups underconfident (0.77 at n_g=25-150). The `_BERNOULLI_BLUP_VAR_INFLATION = 1.5` is a gross inflation that helps small groups
but overcorrects large groups.

Bernoulli Improvement Paths
-----------------------------

**Priority 1 — Prior-regularized IRLS β₀ (HIGH impact on FFX)**

Replace `irlsBernoulli` (unregularized Newton-IRLS) with a prior-penalized variant.
Add a ridge penalty proportional to the FFX prior: diagonal penalty `1/τ_ffx_j²` on
each predictor, centering at `ν_ffx`. Modified Newton step:

    H_reg = X'WX + diag(1/τ²)   where τ = tau_ffx per predictor
    grad_reg = X'(y − μ) − (β − ν_ffx) / τ²

At low n/d (the worst-case regime), the regularized IRLS is the main estimator and
GLS refinement becomes a correction. At high n/d, the ridge penalty is negligible
relative to `X'WX` → fallback to current unregularized behavior.

Requires: pass `nu_ffx`, `tau_ffx` through `_initialFixedEffects` → `irlsBernoulli`
→ `_initialPqlState` → `_lmmGlmm` → `lmmBernoulli`.

Acceptance: reduce FFX NRMSE by ≥ 20% at large-b-sampled (current: 2.80) without
regressing small-b-mixed (current: 0.79). Also check sRFX and BLUP for regressions.

Do NOT attempt if `tau_ffx` is not available in the batch (forward as optional, skip
penalty when None).

**Priority 2 — Laplace-MAP σ_rfx refinement (MEDIUM impact on sRFX)**

Analogous to `refineNormalMapSrfx` for Normal. After the main PQL passes, run a
gradient-based MAP optimization of `log σ_rfx` using:
- Laplace log-marginal as likelihood proxy: `log p_Lap ≈ Σ_g[log p(y_g|β,b̂_g) +
  log p(b̂_g|Ψ) − 0.5 log|H_g|]` (all already computed by the PQL pass)
- Prior log-probability `log p(σ_rfx | τ_rfx)` from hyperparameters
- Final rerun of one PQL pass with refined Ψ = diag(σ_rfx_map²)

This is more complex than Normal MAP because the Laplace log-marginal depends on the
Newton solution b̂_g and H_g, which themselves depend on Ψ. Options:
(a) Use the current PQL outputs as fixed-point approximation (ignore gradient through
    b̂_g, H_g — treat as constants). Cheaper; may underestimate gradient.
(b) Differentiate through the Newton solve (full gradient). Expensive but exact.

Start with option (a): fixed-point MAP. If σ_rfx gradient direction is consistent
with the observed bias pattern (positive gradient for low estimates, negative for
high), this is likely sufficient.

Acceptance: reduce sRFX NRMSE by ≥ 10% at large-b-sampled (current: 0.95) with no
FFX or BLUP regressions on the required suite.

Do not implement if option (a) is unstable (e.g., if σ_rfx MAP contradicts Laplace
M-step direction on > 20% of datasets at small-b-mixed).

**Priority 3 — Beta blend for BLUP residuals (LOW impact, quick)**

Apply the Normal-path beta blend technique to Bernoulli final BLUP residuals. In the
final PQL pass, compute BLUP residuals with a blend of GLS and pooled-IRLS beta:

    beta_for_blup = alpha * beta_gls + (1 − alpha) * beta_0

where alpha ≤ 0.65 for low-d, ≤ 0.75 for high-d (analogous to Normal path). Only
the BLUP estimate changes; `beta_est` (reported fixed effects) is unchanged.

Expected gain: small (5–10% at small-medium), possibly neutral or negative at
large-huge (beta_0 from pooled GLM is also poor there). Run oracle ablation to
verify before implementing.

Acceptance target: no regressions on any dataset × partition cell. A small BLUP
improvement at small-b-mixed is sufficient to justify.

**Priority 4 — blup_var calibration tuning (LOW priority)**

`_BERNOULLI_BLUP_VAR_INFLATION = 1.5` overcorrects large groups (ratio 0.77 at
n_g=25-150) while being marginal at small groups (1.31 at n_g=5-9). A group-size-
dependent inflation (e.g., 1.0 + C/n_g with C tuned on the small dataset) would
improve calibration without changing point estimates. Defer until priorities 1–3 are
stable, as blup_var is sensitive to changes in Ψ.

Acceptance Criteria (Bernoulli)
---------------------------------

A Bernoulli GLMM change should be considered only if it improves at least one
primary metric on the required suite without material regressions:

- Target FFX improvement ≥ 15% at large-b or huge-b (the high-d failure modes).
- Target sRFX improvement ≥ 10% at any required cell.
- BLUP improvement must not regress FFX or sRFX.
- Compare against current PQL baseline (no `map_refine` applies for Bernoulli).
- Changes that only improve a narrow bin (single bg_df value or n/d quartile) are
  experiments only — not production unless globally non-harmful.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
uv run python experiments/analytical/glmm_required_benchmark.py --family b
uv run python experiments/analytical/glmm_raw_diagnostic.py
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-n-mixed
uv run python experiments/analytical/glmm_error_analysis.py --data-id small-b-mixed
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
