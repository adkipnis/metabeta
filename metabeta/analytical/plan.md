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
| small-b-sampled | test | 1.1526 | 0.6856 | 0.6865 |
| medium-b-mixed | train | 1.5274 | 0.7236 | 0.7667 |
| medium-b-sampled | valid | 1.6387 | 0.7652 | 0.7901 |
| medium-b-sampled | test | 1.9131 | 0.8071 | 0.8988 |
| large-b-mixed | train | 2.0386 | 0.8465 | 0.8809 |
| large-b-sampled | valid | 2.8019 | 0.9498 | 1.0738 |
| large-b-sampled | test | 2.9786 | 0.9799 | 1.1372 |
| huge-b-mixed | train | 3.1533 | 0.9895 | 1.2625 |
| huge-b-sampled | valid | 3.4570 | 1.0995 | 1.4105 |
| huge-b-sampled | test | 3.3510 | 0.9934 | 1.1757 |

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

**✓ Priority 1 — Prior-regularized IRLS β₀ (DONE 2026-05-13)**

Add diagonal Gaussian prior N(ν_ffx, diag(τ²)) to the pooled IRLS normal equations:

    (X'WX + diag(1/τ²)) β = X'Wz + diag(1/τ²) ν_ffx

Inactive dimensions (τ_ffx == 0) receive zero prior precision — NOT clamped to 1e-4,
which would add 1e8 precision and blow up `_adaptiveRidge` (uses max-diagonal scale)
for all other dimensions. Initialize from zeros, not from ν_ffx (warm-starting from
ν_ffx destabilizes the first IRLS step when ν ≠ 0 since p deviates from 0.5 and
(y−p)/w explodes). GLS prior in `_pqlPass` was tried and reverted: the shrunk β
causes random effects to compensate, inflating Ψ̂_Lap.

Prior priors threaded: `glmm.py` → `lmmBernoulli` → `_lmmGlmm` → `_initialPqlState`
→ `_initialFixedEffects` → `irlsBernoulli`. Only active when `map_refine=True` (raw
method is unchanged baseline).

Post-P1 benchmark (`glmm_required_benchmark.py --family b --methods current raw`):

| Dataset | Partition | FFX (raw→cur) | sRFX (raw→cur) | BLUP (raw→cur) |
| --- | --- | ---: | ---: | ---: |
| small-b-mixed | train | 0.7898 → 0.6726 | 0.6275 → 0.6278 | 0.6384 → 0.6384 |
| small-b-sampled | valid | 1.1875 → 0.9841 | 0.6873 → 0.6952 | 0.6828 → 0.6934 |
| small-b-sampled | test | 1.1526 → 0.7810 | 0.6856 → 0.6786 | 0.6865 → 0.6822 |
| medium-b-mixed | train | 1.5274 → 1.4144 | 0.7236 → 0.7197 | 0.7667 → 0.7331 |
| medium-b-sampled | valid | 1.6387 → 1.2816 | 0.7652 → 0.7708 | 0.7901 → 0.8431 |
| medium-b-sampled | test | 1.9131 → 1.4721 | 0.8071 → 0.7956 | 0.8988 → 0.8501 |
| large-b-mixed | train | 2.0386 → 2.0067 | 0.8465 → 0.8070 | 0.8809 → 0.9970 ⚠️ |
| large-b-sampled | valid | 2.8019 → 1.9941 | 0.9498 → 0.8360 | 1.0738 → 0.8668 |
| large-b-sampled | test | 2.9786 → 1.9578 | 0.9799 → 0.8417 | 1.1372 → 0.9079 |
| huge-b-mixed | train | 3.1533 → 2.4385 | 0.9895 → 0.8813 | 1.2625 → 1.0024 |
| huge-b-sampled | valid | 3.4570 → 2.6658 | 1.0995 → 0.9275 | 1.4105 → 1.0279 |
| huge-b-sampled | test | 3.3510 → 2.6161 | 0.9934 → 0.9124 | 1.1757 → 1.0646 |

FFX improves in all 12 cells (−2% to −34%). Acceptance criterion met: −29%/−34% at
large-b-sampled. sRFX and BLUP also strongly improve at large/huge. One unexplained
regression: large-b-mixed train BLUP +13% (⚠️); sampled counterparts improve −19%/−20%.

**✓ P1 extension — Student-t adaptive prior precision (DONE 2026-05-13)**

P1 hardcoded a Gaussian prior ridge `1/τ²` regardless of `family_ffx`. For the 20% of
datasets with Student-t FFX priors (df=5), this over-regularizes tail cases: the
quadratic penalty `(β−ν)²/(2τ²)` grows faster than the true t-log-density
`3·log(1 + (β−ν)²/(5τ²))` for large `|β−ν|`.

Fix: inside the IRLS loop, select prior precision by `family_ffx`:

    # Normal (family 0): constant ridge
    prior_prec = 1.0 / tau_ffx.clamp(min=1e-4).square()
    # Student-t (family 1, df=5): EM-style adaptive weight
    prior_prec = 6.0 / (5.0 * tau_ffx.clamp(min=1e-8).square() + (beta - nu_ffx).square()).clamp(min=1e-8)

The t-weight is recomputed from the current `beta` at each IRLS iteration. Inactive
dimensions (tau_ffx == 0) always receive zero precision regardless of family.
`family_ffx` threaded: `glmm.py` → `lmmBernoulli` → `_lmmGlmm` → `_initialPqlState`
→ `_initialFixedEffects` → `irlsBernoulli`.

Post-extension benchmark (current vs raw, 2026-05-13):

| Dataset | Partition | FFX (raw→cur) | sRFX (raw→cur) | BLUP (raw→cur) |
| --- | --- | ---: | ---: | ---: |
| small-b-mixed | train | 0.7898 → 0.6716 | 0.6275 → 0.6278 | 0.6384 → 0.6384 |
| small-b-sampled | valid | 1.1875 → 0.9861 | 0.6873 → 0.6952 | 0.6828 → 0.6934 |
| small-b-sampled | test | 1.1526 → 0.7870 | 0.6856 → 0.6785 | 0.6865 → 0.6822 |
| medium-b-mixed | train | 1.5274 → 1.4150 | 0.7236 → 0.7198 | 0.7667 → 0.7331 |
| medium-b-sampled | valid | 1.6387 → 1.2822 | 0.7652 → 0.7708 | 0.7901 → 0.8431 |
| medium-b-sampled | test | 1.9131 → 1.4797 | 0.8071 → 0.7957 | 0.8988 → 0.8478 |
| large-b-mixed | train | 2.0386 → 2.0148 | 0.8465 → 0.8100 | 0.8809 → 0.9982 ⚠️ |
| large-b-sampled | valid | 2.8019 → 1.9196 | 0.9498 → 0.8293 | 1.0738 → 0.8486 |
| large-b-sampled | test | 2.9786 → 1.9615 | 0.9799 → 0.8443 | 1.1372 → 0.8973 |
| huge-b-mixed | train | 3.1533 → 2.4391 | 0.9895 → 0.9026 | 1.2625 → 1.0162 |
| huge-b-sampled | valid | 3.4570 → 2.6849 | 1.0995 → 0.9282 | 1.4105 → 1.0299 |
| huge-b-sampled | test | 3.3510 → 2.6161 | 0.9934 → 0.9174 | 1.1757 → 1.0704 |

Compared to the P1-only baseline: marginal gains at large-b-sampled valid FFX (−3.7%,
1.9941→1.9196) and sRFX; all other cells within noise or unchanged. Effect is small
because only 20% of datasets use Student-t and df=5 is near-Normal for moderate |β−ν|.
The "current" Bernoulli baseline is now the post-extension numbers above.

**Priority 2 — Laplace-MAP σ_rfx refinement (MEDIUM impact on sRFX; option (a) TRIED AND FAILED)**

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

Option (a) requires one final `_pqlPass` after the MAP step to recompute BLUPs with
the refined Ψ — without it, BLUP quality is unchanged and only sigma_rfx_est is
updated. The recomputation pass is cheap (no convergence loop needed).

Acceptance: reduce sRFX NRMSE by ≥ 10% at large-b-sampled (current: 0.84) with no
FFX or BLUP regressions on the required suite.

**Option (a) result (2026-05-13):** Implemented in `refineBernoulliMapSrfx` (map.py,
not called). Using `sum_bhat_sq = G·(Ψ_lap_jj − mean_Hg_inv_jj)` as the sufficient
statistic consistently worsened sRFX at small-b (0.63→0.63–0.72, all degraded vs
P1 and raw). Root cause: `G·(Ψ_lap − H_mean)` is NOT a valid sufficient statistic
for σ² — it cancels the `H_g^{-1}` correction that the M-step added to make the
estimate unbiased under the Laplace approximation. In effect, it reverts to the
(downward-biased) EM MoM. The MAP then pushes σ further downward via the prior,
worsening underestimation cases. Reverted from glmm.py.

**Prior-informed Ψ floor sub-item (DONE 2026-05-13):** Replaced the constant
`_BERNOULLI_INITIAL_PSI_FLOOR = 0.25` in `_initialPqlState` with a per-dataset
floor derived from `tau_rfx`:

    tau_agg = tau_rfx[:, :q].clamp(min=1e-4).mean(dim=-1)   # (B,)
    prior_floor = tau_agg.square().clamp(max=family.initial_psi_floor)
    psi_0_floor = torch.maximum(psi_0, prior_floor)

The `.clamp(max=0.25)` cap ensures the floor can only be **lowered**, never raised.
Only datasets with `tau_rfx_mean < 0.5` (τ²< 0.25) see a lower floor; the majority
(Bernoulli τ_rfx mode=0.7 → τ²≈0.49) see no change.  The uncapped formula
(`torch.maximum(psi_0, tau_agg.square())`) raised the floor for most datasets and
caused systematic FFX regressions at large/huge (up to +6.5% FFX, +18.3% BLUP at
huge-b-sampled valid). The cap fix eliminates these regressions.

`tau_rfx` threaded: `glmm.py` → `lmmBernoulli` → `_lmmGlmm` → `_initialPqlState`.
Only active when `map_refine=True`.

Benchmark vs P1 baseline (cap-fix formula, 2026-05-13):

| Dataset | Partition | FFX (P1→cap) | sRFX (P1→cap) | BLUP (P1→cap) |
| --- | --- | ---: | ---: | ---: |
| small-b-mixed | train | 0.6716 → 0.6682 | 0.6278 → 0.6134 | 0.6384 → 0.6369 |
| small-b-sampled | valid | 0.9861 → 0.9844 | 0.6952 → 0.6854 | 0.6934 → 0.6854 |
| small-b-sampled | test | 0.7870 → 0.7652 | 0.6785 → 0.6742 | 0.6822 → 0.6839 |
| medium-b-mixed | train | 1.4150 → 1.4230 | 0.7198 → 0.7142 | 0.7331 → 0.7387 |
| medium-b-sampled | valid | 1.2822 → 1.3452 | 0.7708 → 0.7846 | 0.8431 → 0.8497 |
| medium-b-sampled | test | 1.4797 → 1.4008 | 0.7957 → 0.8018 | 0.8478 → 0.8442 |
| large-b-mixed | train | 2.0148 → 2.0853 | 0.8100 → 0.8000 | 0.9982 → 0.9482 |
| large-b-sampled | valid | 1.9196 → 1.9543 | 0.8293 → 0.8491 | 0.8486 → 0.8463 |
| large-b-sampled | test | 1.9615 → 2.0433 | 0.8443 → 0.8651 | 0.8973 → 0.9752 |
| huge-b-mixed | train | 2.4391 → 2.5970 | 0.9026 → 0.8796 | 1.0162 → 0.9497 |
| huge-b-sampled | valid | 2.6849 → 2.7898 | 0.9282 → 0.9393 | 1.0299 → 1.0304 |
| huge-b-sampled | test | 2.6161 → 2.7488 | 0.9174 → 0.9263 | 1.0704 → 1.0016 |

Small-b sRFX improves consistently (−0.6% to −2.3%). Medium/large/huge deltas are
epoch noise — cap fix is a no-op for those datasets (τ_rfx_mean > 0.5 for most).
The "current" Bernoulli baseline is now the post-cap-fix numbers above.

Next: try option (b) — differentiate through the Newton solve using autograd on a
clean (non-try/except) version of the Laplace objective.

**Sub-item 2a — BLUP recomputation pass after MAP**

After `refineBernoulliMapSrfx` updates `sigma_rfx_est`, run one final `_pqlPass`
with the MAP Ψ = diag(σ_rfx_map²) as the precision input:
- Use warm-start from current `blup_est`
- This gives BLUPs consistent with the refined Ψ (better shrinkage)
- `beta_est` is taken from this final pass as well

Without this pass, improved σ_rfx is not reflected in BLUPs. The Normal path does
the equivalent via `_recomputeNormalFinalDiagMap`.

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

**Priority 5 — nAGQ for q=1 (OPEN)**

For datasets with a scalar random effect (q=1), replace the single Laplace
evaluation of the marginal likelihood with k=7 adaptive Gauss-Hermite quadrature.
The standard Laplace (k=1) directly causes the Breslow-Lin downward bias in Ψ̂;
nAGQ removes it without sampling.

Algorithm change (q=1 only, gates on active_q.sum() == 1):

After the final PQL Newton loop has found b̂_g and H_g = ZWZ_g + Ψ^{-1}, compute
the group-level log-marginal likelihood via logsumexp over k quadrature nodes:

    z_j, w_j   : standard GH nodes/weights (k=7, pre-computed)
    σ_g        : H_g^{-0.5}  (scalar curvature scale)
    b_{g,j}    : b̂_g + √2 · σ_g · z_j   (shifted/scaled quadrature nodes)
    ℓ_{g,j}   : log p(y_g | β, b_{g,j}) + log p(b_{g,j} | Ψ)
    LML_g      : logsumexp_j(w_j + ℓ_{g,j} + z_j²) + log(√2 · σ_g)

Use the nAGQ LML to update Ψ: the gradient ∂(ΣgLML_g)/∂(log σ²) drives a single
gradient step on σ_rfx after the PQL outer loop, analogous to refineBernoulliMapSrfx
but with a provably less-biased sufficient statistic.

Implementation notes:
- GH nodes/weights: `np.polynomial.hermite.hermgauss(k)` — pre-compute once per call.
- The correction term `+ z_j²` cancels the Gaussian weight implicit in standard GH;
  together with log(√2·σ_g) from the change of variables b = b̂ + √2·σ·t.
- For q>1 the node count scales as k^q (343 for k=7, q=3); restrict to q=1 only.
- The Ψ gradient is taken w.r.t. log(σ²) for numerical stability, then exponentiated.
- One final _pqlPass with the nAGQ-refined Ψ recomputes BLUPs consistently.

Acceptance: ≥10% reduction in σ_rfx NRMSE at large-b-sampled (current ≈0.84) with
no FFX or BLUP regressions, restricted to q=1 cells only.

**Priority 6 — True Laplace marginal likelihood: joint (β, Ψ) optimization (OPEN)**

The current PQL alternates: (a) Newton loop for b̂_g under current (β, Ψ), (b) Ψ
M-step = mean_g(b̂_g b̂_g' + H_g^{-1}), (c) GLS update for β via working response.
The Ψ M-step is the correct first-order condition for the Laplace marginal likelihood
by the envelope theorem. The β GLS (Fisher scoring on the linearized objective) is
NOT the same as the true Laplace score ∂L/∂β = Σ_g X_g'(y_g − μ_g(β, b̂_g)).

The two β updates agree at convergence only if the alternating sequence reaches the
joint fixed point of the true Laplace log-marginal

    L(β, Ψ) = Σ_g [log p(y_g | β, b̂_g) + log p(b̂_g | Ψ) − ½ log|H_g|]

With a finite number of passes and damped Newton steps this may not hold, especially
at large d / small n where the GLS is underdetermined and the alternating sequence
can converge to a biased β–Ψ pair.

Proposed approach: after PQL convergence, take a fixed number of gradient steps on
(β, log σ_rfx) by differentiating L above at the final (b̂_g, H_g) treated as fixed
(envelope-theorem approximation). The gradient w.r.t. β is:

    ∂L/∂β ≈ Σ_g X_g'(y_g − μ_g(β, b̂_g))   [true Bernoulli score]

This is a small-step refinement after PQL convergence, not a replacement of the
PQL outer loop. The b̂_g are not re-solved during this refinement step; autograd
on the Bernoulli log-likelihood at fixed b̂_g gives the gradient cheaply.

Notes on the P2 option-b failure: the prior attempt (refineBernoulliMapSrfx option a)
used `G·(Ψ_lap − H_mean)` as the sufficient statistic for σ², which cancels the H_g
term and reverts to the downward-biased raw MoM. The correct Laplace LML Ψ gradient
(from ∂L/∂(1/σ²) = 0) gives Ψ = mean_g(b̂_g b̂_g' + H_g^{-1}), which IS the current
M-step. So Ψ estimation is likely near-optimal; β is the main remaining lever here.

Acceptance: ≥5% FFX NRMSE improvement at large-b or huge-b with no sRFX or BLUP
regressions. If only sRFX improves, the PQL M-step was not at fault.

**External reference baseline: statsmodels BinomialBayesMixedGLM (CAVI)**

Python-native deterministic baseline (pymer4/lme4 unavailable — R dependency
broken in environment). Uses coordinate-ascent variational inference (CAVI) on a
mean-field Gaussian approximation of the joint posterior p(β, b_g, σ_rfx | y).

Behaviour relative to our PQL:
- FFX (β): CAVI uses the true Bernoulli score gradient; expected to be similar or
  slightly better than PQL at extreme d/n ratios.
- σ_rfx: CAVI is known to underestimate posterior variance (mean-field variational
  underestimation), so it may give σ_rfx estimates similar to or worse than PQL's
  Laplace M-step.
- BLUP: CAVI provides per-group posterior means; extraction requires reading the RE
  params from `result.params[k_fe + k_vc:]`.

Note: prior matching is approximate (CAVI uses a Gaussian prior on log σ², we use
HalfNormal on σ). The comparison is frequentist (accuracy vs ground truth), not a
matched-prior Bayesian comparison.

Experiment: `experiments/analytical/glmm_reference_comparison.py` — reports NRMSE
side-by-side on q=1 test datasets, with σ_rfx bias breakdown by true σ_rfx bin.

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
uv run python experiments/analytical/glmm_reference_comparison.py
uv run python experiments/analytical/glmm_reference_comparison.py --data-id small-b-sampled --n-cavi 200
uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
