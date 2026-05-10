Plan
====

Last updated: 2026-05-09, after final I9 output calibration tuning.

Current state
-------------

The Gaussian analytical estimator is in a stable I9 output-calibrated state:

- BLUP accuracy is mainly from output-local beta handling:
  - I5/I6: compute final BLUP residuals with a blend toward pooled OLS.
  - I7: make that blend active-d adaptive:
    - alpha `1.00` for active d <= 4
    - alpha `0.65` for active d 5-8
    - alpha `0.75` for active d > 8
- sRFX accuracy is improved by floor-aware handling:
  - I8: lower joint diagonal MoM floor signal from `0.5` to `0.45`.
  - I9: keep runtime floors unchanged, but calibrate reported sigma(RFX) after BLUPs:
    - q > 2 and floor-hit: `sigma_rfx_est = 0.55 * sqrt(Psi_diag)`.
    - q = 2, floor-hit, and `Psi_diag / psi_diag_floor <= 0.68`:
      `sigma_rfx_est = 0.85 * sqrt(Psi_diag)`.
- Runtime floors remain important for GLS/EM stability. Do not lower them again without a
  stronger observable gate.
- `_remlNewtonStep` still exists in `metabeta/analytical/normal.py` but is unused.

Current benchmark
-----------------

Required-suite result:

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.6115 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.6302 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6519 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.4516 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.5321 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.5989 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.4630 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.5401 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.5282 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.4465 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.6549 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5701 | 0.1828 | 0.5167 |

Compared with I8, FFX, sEps, and BLUP are unchanged. sRFX improves on every row. The
largest remaining required sRFX errors are `huge-n-sampled/valid` at `0.6549` and
`small-n-sampled/test` at `0.6519`.

What worked
-----------

- Output-local beta changes worked for BLUP because they avoided changing reported
  `beta_est`, Psi, or sigma_eps.
- I7 active-d adaptive alpha improved small-n BLUP while keeping medium rows inside the
  regression budget.
- A mild I8 runtime diagonal floor reduction helped sRFX, but only at `0.45`.
- I9 output-only sigma calibration worked because it changed only reported marginal
  random-effect scale and returned `Psi`, after GLS/EM/BLUP/sigma_eps were computed.
- The best accepted I9 schedules were:
  - q > 2 floor-hit sigma factor `0.55`.
  - q = 2 low floor-ratio sigma factor `0.85`.

What failed
-----------

- `beta_wg` in MoM residuals: medium-n-mixed FFX blew up.
- Reported-beta blending: BLUP improved but FFX regressed.
- More EM or psi_df-gated EM: converged to biased or unstable fixed points.
- REML gates tried so far: no BLUP gain with too much sRFX risk.
- Broad runtime floor reductions:
  - `0.25 * psi_diag_signal`: failed `medium-n-sampled/test` BLUP and FFX.
  - `0.375 * psi_diag_signal`: failed mixed rows badly.
  - non-full diagonal-MoM floors `0.25`, `0.35`, `0.40`: improved sRFX locally but moved
    FFX or BLUP.
  - lower fallback/component runtime floors: worsened sRFX and sEps on sampled rows.
- I9 diagnostic rejected:
  - q > 2 factors `0.80` and `0.85`: too conservative after the factor sweep.
  - weak-path-only q > 1 factors `0.85` and `0.90`: worsened all required rows.
  - cap-floor factor `0.75`: negligible gains with more losses than wins.

Guardrails
----------

For each estimator patch, run:

```bash
uv run python experiments/analytical/glmm_required_benchmark.py
```

Required suite:

- mixed datasets: `small|medium|large|huge-n-mixed`, train epochs 1 and 2.
- sampled datasets: `small|medium|large|huge-n-sampled`, valid and test.

Keep rules:

- Keep I7 BLUP beta schedule fixed unless a new diagnostic directly targets BLUP.
- Use only observable numerical gates; no truth, dataset family, or partition gates.
- For sRFX patches, FFX, sEps, and BLUP should stay unchanged or nearly unchanged.
- No BLUP row may regress by more than 3% versus the current accepted baseline.
- Revert immediately after a clear medium/large/huge regression.

Ranked remaining paths
----------------------

1. **Check whether the final I9 factor is over-tuned.**
   Run the updated calibration diagnostic with the q = 2 gate fixed and compare nearby
   q > 2 factors (`0.50`, `0.55`, `0.60`). Patch only if a candidate improves mean sRFX
   without worsening the current max rows.

2. **q = 1 floor-hit calibration.**
   q = 1 floor-hit components remain biased, but their global sRFX is lower than q = 2/q
   > 2 floor-hit components. Test only output-only gates and avoid changing runtime
   floors.

3. **Low floor-ratio residual calibration.**
   The low floor-ratio bin remains high-error. It may still isolate over-flooring better
   than path labels, but q = 2 already consumed the cleanest version of this signal.

4. **Weak-path floor-hit calibration with better gates.**
   `diag_mom`, `component_diag`, and fallback floor-hit components remain high-error, but
   broad weak-path output schedules failed. Revisit only with q/floor-ratio gates.

5. **Eigencap-hit component handling.**
   Low promise for the required suite. Cap-hit rows are rare, and the cap-floor output
   schedule did not help.

6. **Off-diagonal correlation shrinkage.**
   Lower priority for current metrics. Correlation error may matter for returned `Psi`,
   BLUP covariance, or downstream NN inputs more than for required sRFX.

7. **Runtime EM/M-step/REML changes.**
   Lowest priority. Prior attempts showed biased fixed points or collateral regressions.

Next step
---------

Run `glmm_i9_calibration_diagnostic.py` with the current schedule as baseline, then test
only one narrow next patch if it improves the required suite without moving FFX, sEps, or
BLUP. The best next diagnostic target is q = 1 output-only floor calibration; runtime
floor changes should remain off the table.
