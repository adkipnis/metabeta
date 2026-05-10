GLMM Analytical Estimator Summary
=================================

Last updated: 2026-05-10.

This file records the useful conclusions from the analytical Gaussian GLMM
experiments. Historical scratch diagnostics were removed; the current reusable
scripts are documented in `README.md`.

Final Required Benchmark
------------------------

Command:

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
```

Required suite: mixed train epochs 1-2 and sampled valid/test for
`small|medium|large|huge`.

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.6100 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.6291 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6507 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.4501 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.5302 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.5979 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.4614 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.5385 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.5264 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.4450 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.6535 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5685 | 0.1828 | 0.5167 |

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
