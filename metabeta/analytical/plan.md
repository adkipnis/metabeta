Plan
====

Last updated: 2026-05-09, after I9 floor-pinned output calibration.

Current state
-------------

The Gaussian analytical estimator is in a stable I9 state:

- BLUP accuracy was mainly fixed by output-local beta handling:
  - I5/I6: compute final BLUP residuals with a blend toward pooled OLS.
  - I7: make that blend active-d adaptive:
    - alpha `1.00` for active d <= 4
    - alpha `0.65` for active d 5-8
    - alpha `0.75` for active d > 8
- sRFX accuracy is now improved by floor-aware handling:
  - I8: lower joint diagonal MoM floor signal from `0.5` to `0.45`.
  - I9: output-only calibration for floor-pinned rows with active q > 2:
    `sigma_rfx_est = 0.8 * sqrt(Psi_diag)`.
- Runtime floors remain important for GLS/EM stability. Do not lower them again without a
  stronger observable gate.
- `_remlNewtonStep` still exists in `metabeta/analytical/normal.py` but is unused.

Current I9 benchmark
--------------------

Required-suite result:

| Dataset | Partition | FFX | sRFX | sEps | BLUP |
| --- | --- | ---: | ---: | ---: | ---: |
| small-n-mixed | train | 0.2250 | 0.6353 | 0.0839 | 0.3625 |
| small-n-sampled | valid | 0.1553 | 0.6313 | 0.1036 | 0.4249 |
| small-n-sampled | test | 0.1685 | 0.6623 | 0.1002 | 0.4207 |
| medium-n-mixed | train | 0.1459 | 0.5065 | 0.0671 | 0.3714 |
| medium-n-sampled | valid | 0.3625 | 0.5312 | 0.0978 | 0.4920 |
| medium-n-sampled | test | 0.2437 | 0.6003 | 0.1030 | 0.4788 |
| large-n-mixed | train | 0.2700 | 0.4736 | 0.0724 | 0.3641 |
| large-n-sampled | valid | 0.3658 | 0.5484 | 0.1105 | 0.4605 |
| large-n-sampled | test | 0.4627 | 0.5589 | 0.1082 | 0.4803 |
| huge-n-mixed | train | 0.3069 | 0.4655 | 0.0649 | 0.3965 |
| huge-n-sampled | valid | 0.4204 | 0.7053 | 0.1644 | 0.5086 |
| huge-n-sampled | test | 0.4661 | 0.5740 | 0.1828 | 0.5167 |

Compared with I8, I9 preserves every FFX, sEps, and BLUP row; it leaves small rows
unchanged and improves all medium/large/huge sRFX rows. The largest remaining required
sRFX error is `huge-n-sampled/valid` at `0.7053`.

What worked
-----------

- Output-local beta changes worked for BLUP because they avoided changing reported
  `beta_est`, Psi, or sigma_eps.
- I7 active-d adaptive alpha improved small-n BLUP while keeping medium rows inside the
  regression budget.
- A mild I8 runtime diagonal floor reduction helped sRFX, but only at `0.45`.
- I9 output-only floor-pinned sigma calibration worked because it changed only reported
  marginal random-effect scale, after GLS/EM/BLUP/sigma_eps were already computed.

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

1. **Refine I9 output calibration.**
   Most promising because I9 gave clean sRFX gains with zero FFX/sEps/BLUP movement.
   The next question is whether the q > 2 gate and `0.8` sigma factor can be improved
   using observable bins: active q, path, `G_mom`, floor ratio, cap status, and active d.

2. **Output-only calibration for weak floor-hit paths.**
   Promising but narrower. `diag_mom`, `component_diag`, and fallback floor-hit components
   remain high-error, while internal floor changes were unstable. Test output-only
   calibration first, not runtime floor changes.

3. **Eigencap-hit component handling.**
   Moderate promise. Cap-hit rows are rare, so benchmark leverage is limited, but their
   error is high. Prefer output-only cap calibration or a diagnostic before changing the
   eigencap used by GLS/EM.

4. **Off-diagonal correlation shrinkage.**
   Lower priority for the current metrics. Correlation error remains high, but sRFX is a
   diagonal scale metric. This may matter more for returned `Psi`, BLUP covariance, or
   downstream NN inputs than for the required benchmark.

5. **Runtime EM/M-step/REML changes.**
   Lowest priority. Prior attempts showed biased fixed points or collateral regressions.
   Reopen only with a new diagnostic that explains why the previous failures would be
   avoided.

Next step
---------

Build an I9 calibration diagnostic before patching again.

1. Extend or add an analytical experiment that replays the I9 output calibration and
   reports candidate output-only schedules without changing estimator internals.
2. Bin floor-pinned rows by:
   - active q: `1`, `2`, `3+`
   - MoM path: `full_mom`, `diag_mom`, `component_diag`, `fallback`
   - `G_mom` quantiles
   - floor ratio: `Psi_diag / psi_diag_floor`
   - cap-hit status
3. Compare candidate schedules:
   - current I9: q > 2, sigma factor `0.8`
   - q > 1 with milder factor
   - q > 2 with per-path factors
   - weak-path-only output calibration
   - cap-hit-only output calibration
4. Patch only the best single schedule if it improves required-suite sRFX without moving
   FFX, sEps, or BLUP. Otherwise leave I9 unchanged.
