GLMM Analytical Estimator Summary
=================================

Last updated: 2026-05-10.

This file records the useful conclusions from the analytical Gaussian GLMM
experiments. Historical scratch diagnostics were removed; the current reusable
scripts are documented in `README.md`.

Current Production MAP Benchmark
--------------------------------

Command:

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
```

Required suite: mixed train epochs 1-2 and sampled valid/test for
`small|medium|large|huge`. These rows are the benchmark baseline for the current
production marginal MAP sRFX path. The production MAP step replaces only
`sigma_rfx_est` and the `Psi` diagonal; FFX, sEps, and BLUP remain MoM/EM-derived.

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.5650 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.5819 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6242 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.3700 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.4671 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.5129 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.3849 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.4679 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.4677 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.3932 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.5911 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5050 | 0.1828 | 0.5167 |

Accepted Fixes
--------------

- **BLUP variance floor**: adding `Psi_diag / G_mom` to `blup_var` reduced severe
  overconfidence for large groups. It improves uncertainty calibration only; BLUP
  point estimates do not change.
- **Delta-method cap loosened**: changing the denominator clamp from 4 to 2 allows
  more `blup_var` inflation in very low-df cases. It had little aggregate effect
  but did not regress the required suite.
- **Off-diagonal Psi shrinkage**: shrinking correlations with `G / (G + 5)` makes
  returned covariance less noisy for small group counts. It does not affect
  marginal sRFX.
- **Component-wise MoM reliability gate**: requiring at least 5 valid groups before
  using component-wise diagonal signals prevented noisy high-q components from
  inflating `psi_eig_cap`. This was the first large BLUP/FFX improvement.
- **Adaptive M-step BLUP winsorization**: clipping EM M-step BLUPs at
  `10 * sqrt(Psi_diag)` prevented extreme groups from dominating `Psi_em`. The
  chosen multiplier gave broad BLUP gains with acceptable tradeoffs.
- **Output-local BLUP beta blend**: computing final BLUP residuals with an
  active-d adaptive blend toward pooled OLS improved BLUP without changing reported
  `beta_est`, `sigma_eps_est`, or runtime Psi.
- **Mild runtime diagonal-floor reduction**: lowering the joint diagonal MoM floor
  signal from `0.50` to `0.45` helped sRFX. Broader floor reductions were unstable.
- **Output-only sRFX calibration**: reported marginal random-effect scale is
  calibrated after GLS/EM/BLUP:
  - q = 1, floor-hit, `Psi_diag / psi_diag_floor <= 0.68`: sigma factor `0.70`.
  - q = 2, floor-hit, `Psi_diag / psi_diag_floor <= 0.68`: sigma factor `0.85`.
  - q > 2 and floor-hit: sigma factor `0.55`.

Rejected Fixes
--------------

- **Using `beta_wg` in MoM residuals** improved some residual decomposition but
  caused medium-n mixed FFX blowups.
- **Reported-beta blending** improved BLUP but regressed FFX because it changed the
  public beta estimate.
- **More EM iterations / G_mom-gated EM** helped selected large rows but converged
  other regimes to biased fixed points, especially medium-n and high-d cases.
- **REML/Newton gates** did not provide reliable BLUP gains and carried too much
  sRFX risk.
- **Median-based MoM cap** collapsed per-component caps when only 1-2 groups were
  available, severely underestimating Psi and breaking GLS/BLUP.
- **Aggressive runtime floor reductions** improved some sRFX bins but moved FFX,
  sEps, or BLUP. Runtime floors are stabilizers; only output calibration remained
  clean.
- **Weak-path-only and cap-hit output schedules** were too broad or too rare to
  improve the required suite.

Remaining Weaknesses
--------------------

- The largest remaining sRFX errors are sampled rows, especially
  `huge-n-sampled/valid` and `small-n-sampled/test`.
- Floor-hit components are still biased. Output calibration reduces the reported
  marginal error, but the internal runtime floor remains because lowering it
  destabilized GLS/EM.
- FFX/BLUP improvements from changing runtime estimation appear mostly exhausted
  without a larger estimator redesign; previous structural changes repeatedly
  helped one regime while breaking another.
- Off-diagonal Psi quality is only lightly handled. It is lower priority for the
  current required metrics, which score marginal sRFX and BLUP point estimates.

Future Directions
-----------------

- **Option 1: refined marginal MAP / REML over variance components.** Next
  concrete diagnostic. Optimize variance scales from the MoM/EM initialization
  using the exact Gaussian marginal likelihood already used by the MAP path.
  Expected cost is roughly one extra short Adam loop per batch; expected gain is
  possible sRFX improvement, especially on sampled rows. Do not integrate unless
  it beats the current production MAP baseline without material FFX, sEps, or BLUP
  regressions.
- **Option 2: Laplace uncertainty around the MAP point.** Secondary direction.
  Estimate curvature after MAP for context-feature uncertainty, not as a direct
  point-estimate improvement. Expected cost is higher than Option 1 and expected
  gain is better uncertainty information rather than lower point NRMSE.

VI-family methods, SGLD, SVGD, MH, and full MCMC are not active analytical
estimator plans.

REML/Profile MAP Diagnostic Result
----------------------------------

Command:

```bash
uv run python experiments/analytical/glmm_reml_diagnostic.py
```

The first diagnostic run is promising for sRFX, but remains experiment-only. The
diagnostic keeps FFX and BLUP unchanged; `reml_diag_seps` also moves sEps.

| Dataset | Partition | current sRFX | reml_diag sRFX | reml_diag_seps sRFX |
| --- | --- | ---: | ---: | ---: |
| small-n-mixed | train | 0.5650 | 0.5637 | 0.5638 |
| small-n-sampled | valid | 0.5819 | 0.5751 | 0.5750 |
| small-n-sampled | test | 0.6242 | 0.6245 | 0.6201 |
| medium-n-mixed | train | 0.3700 | 0.3406 | 0.3398 |
| medium-n-sampled | valid | 0.4671 | 0.4478 | 0.4474 |
| medium-n-sampled | test | 0.5129 | 0.4777 | 0.4903 |
| large-n-mixed | train | 0.3849 | 0.3611 | 0.3610 |
| large-n-sampled | valid | 0.4679 | 0.4394 | 0.4403 |
| large-n-sampled | test | 0.4677 | 0.4293 | 0.4280 |
| huge-n-mixed | train | 0.3932 | 0.3826 | 0.3826 |
| huge-n-sampled | valid | 0.5911 | 0.5504 | 0.5512 |
| huge-n-sampled | test | 0.5050 | 0.4829 | 0.4825 |

Measured suite time in this workspace:

- `current`: 444.33 seconds.
- `mom_em`: 104.71 seconds.
- `reml_diag`: 267.92 seconds for the refinement loop; 372.63 seconds with
  MoM/EM initialization.
- `reml_diag_seps`: 276.94 seconds for the refinement loop; 381.65 seconds with
  MoM/EM initialization.

Do not wire this into `glmm.py` until BLUP/GLS recomputation choices are tested
and sEps movement from the sigma(Eps) variant is judged acceptable.

REML Breakdown Diagnostic Result
--------------------------------

Command:

```bash
uv run python experiments/analytical/glmm_reml_diagnostic.py --breakdown
```

The binned diagnostic confirms that the aggregate `reml_diag` win is not uniform.
It is strongest where the current MAP path shrinks MoM/EM scale, for multi-component
random effects, and for small true random-effect scales. It slightly regresses q=1
overall. Clamp rate is effectively zero, so bounds are not driving the result.

Selected breakdown rows:

| Breakdown | N | current sRFX | reml_diag sRFX | Relative improvement | Fallback rate | Clamp rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 98304 | 0.4898 | 0.4748 | 3.04% | 0.6800 | 0.0003 |
| q=1 | 49941 | 0.4675 | 0.4712 | -0.80% | 0.8621 | 0.0000 |
| q=2 | 25747 | 0.4789 | 0.4679 | 2.31% | 0.6038 | 0.0006 |
| q=3+ | 22616 | 0.5585 | 0.5166 | 7.49% | 0.3647 | 0.0008 |
| map_direction=expand | 77164 | 0.4495 | 0.4482 | 0.30% | 0.6895 | 0.0003 |
| map_direction=shrink | 21106 | 0.6001 | 0.5519 | 8.03% | 0.6452 | 0.0003 |
| true_sigma=<0.25 | 62098 | 0.5632 | 0.5180 | 8.03% | 0.6978 | 0.0004 |
| true_sigma=0.25-0.75 | 29554 | 0.5758 | 0.5477 | 4.88% | 0.6246 | 0.0003 |
| n=2000+ | 12176 | 0.4262 | 0.4358 | -2.26% | 0.6702 | 0.0006 |

The first conservative gate uses REML only for valid, unclamped rows with
`q >= 2` and `n < 2000`; all other rows keep current MAP. On the full required
suite this gives:

| Method | Mean suite sRFX row NRMSE |
| --- | ---: |
| mom_em | 0.5410 |
| current MAP | 0.4898 |
| reml_diag | 0.4748 |
| reml_gated | 0.4745 |

No aggregate required-suite cell has both current MAP and REML worse than MoM/EM,
but row-level cases do exist. The `both_worse_than_mom_rate` is 0.2374 overall,
and it is concentrated in low-change MAP rows (`map_delta=0.1-5pct`: 0.5092)
and current-MAP shrink rows (`map_direction=shrink`: 0.3785). This means MoM/EM
is not a good blanket fallback, but future gates may need a no-refinement branch
for rows where MAP barely changes MoM/EM or where marginal optimization disagrees
with the per-row squared error.

Next decision check: test whether recomputing final GLS/BLUP after gated REML
preserves the current FFX/BLUP stability. The high fallback rate means production
integration needs row-level fallback and explicit logging/metrics.

Statistical Interpretation of Best Method
-----------------------------------------

Best row-weighted sRFX NRMSE by diagnostic bin:

| Case | Best setup | Evidence |
| --- | --- | --- |
| Overall | `reml_gated` | 0.4745 vs raw REML 0.4748 and current MAP 0.4898 |
| q = 1 | current MAP / gated | 0.4675 vs raw REML 0.4712 |
| q = 2 | raw REML | 0.4679 vs gated 0.4692 and current MAP 0.4789 |
| q >= 3 | raw REML | 0.5166 vs gated 0.5187 and current MAP 0.5585 |
| n >= 2000 | current MAP / gated | 0.4262 vs raw REML 0.4358 |
| n < 2000 | raw REML | 0.4992-0.4497 beats current MAP and MoM/EM in both n bins |
| d <= 4 | gated REML | 0.5877 vs raw REML 0.5902 and current MAP 0.5928 |
| d >= 5 | raw REML | 6.0% relative gain over current MAP in both d bins |
| current MAP expands MoM/EM | gated REML | 0.4462 vs raw REML 0.4482 and current MAP 0.4495 |
| current MAP shrinks MoM/EM | raw REML | 0.5519 vs current MAP 0.6001 |
| MAP barely changes MoM/EM (<5%) | MoM/EM | 0.5156 or tie; both refinement paths can be worse |
| true sigma < 0.75 | raw REML | strongest gains: 4.88-8.03% over current MAP |
| true sigma 0.75-1.5 | gated REML | 1.2599 vs raw REML 1.2696 and current MAP 1.2763 |

Likely explanation:

- q = 1 is already well handled by current output calibration, and there is little
  covariance structure for REML to resolve. The REML objective often falls back
  and can add small fixed-beta/profile-likelihood noise when it does not.
- q >= 2 benefits because MoM/EM and output calibration struggle to split scale
  across multiple random-effect components. The marginal objective uses the full
  group likelihood, so it can correct over- or under-shrunk component scales.
- High-n rows expose misspecification in the fixed beta/correlation profile. The
  marginal likelihood is sharper, so a fixed beta or fixed correlation error can
  move sigma(RFX) away from the metric target even when the objective improves.
- When current MAP barely changes MoM/EM, the data/prior signal is weak or already
  balanced. Additional optimization is mostly noise; these are the strongest
  candidates for a future "no refinement" branch.
- When current MAP shrinks MoM/EM, MoM/EM is often overestimating marginal random
  scale. REML usually helps because it directly optimizes the marginal likelihood
  around the MoM/EM initialization rather than applying a fixed output calibration.
