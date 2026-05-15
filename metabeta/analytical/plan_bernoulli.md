Bernoulli GLMM Plan
===================

Last updated: 2026-05-15 (debloated after P14/P15 matched benchmarks)

Goal
----

Build fast, high-accuracy Bernoulli GLMM summaries for downstream hierarchical NPE context.
The analytical estimator does not need to be a full posterior engine; it needs stable,
prior-aware point summaries and uncertainty proxies that condition the NPE well.

Do **not** add a separate amortized correction branch here. `glmm()` outputs are already
passed to the hierarchical NPE, which is the correction mechanism.

Do **not** pursue R-INLA as a backend or full PyTorch INLA as the main branch. Full INLA-style
hyperparameter integration is too expensive for the `~100 ms/dataset` target, especially with
diagonal Ψ up to `q=5` and full Ψ up to 15 hyperparameters. Use INLA concepts, not full INLA.

Current Estimators
------------------

**Default Bernoulli path:** `lmmBernoulli` + `refineBernoulliNagqSrfx` + `refineBernoulliNestedBeta`.
Active when `map_refine=True` and `bernoulli_laplace_eb=False`.

**P14 Laplace-EB path:** `refineBernoulliLaplaceEb`, exposed through:

- `glmm(..., bernoulli_laplace_eb=True)` for all Bernoulli datasets.
- `glmm(..., bernoulli_laplace_eb='auto')` for P15-gated datasets.

P14 optimizes β and diagonal log σ on the true Bernoulli marginal-Laplace target with nested
per-group random-effect modes, priors in the objective, σ continuation, and an objective
acceptance gate against the incoming default path.

P14 knobs:

- `bernoulli_laplace_eb_steps` default `12`
- `bernoulli_laplace_eb_inner` default `4`
- `bernoulli_laplace_eb_final` default `6`
- `bernoulli_laplace_eb_lr` default `0.05`
- `bernoulli_laplace_eb_blup_fallback_beta_jump` default `1.0`, set `None` to disable

P14 diagnostics:

- `laplace_eb_accept`
- `laplace_eb_steps`
- `laplace_eb_target`
- `laplace_eb_base_target`
- `laplace_eb_blup_fallback`
- `laplace_eb_beta_jump`

**P15 auto gate:** currently uses effective `d >= 4`, mean estimated `σ_rfx >= 0.75`, or max
fitted `|η| >= 8`. These are configurable through `bernoulli_laplace_eb_gate_*` kwargs. In
matched medium+ benchmarks, this gate selects 100% of datasets, so it is a small-scale cost
saver rather than a medium/large accuracy selector.

Current Performance
-------------------

Matched first-1000 benchmark, 2026-05-15. `p14_deep` means 20 outer steps, 4 inner mode steps,
and 8 final mode steps. CPU timings are per dataset.

| Dataset | part | Current FFX | P14-all FFX | P14-deep FFX | Current σ | P14-deep σ | Current BLUP | P14-deep BLUP | deep ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2710 | 0.2703 | 0.2687 | 0.5712 | 0.5170 | 0.6176 | 0.6198 | 15.76 |
| small-b-sampled | valid | 0.3186 | 0.3152 | 0.3133 | 0.6041 | 0.5456 | 0.6502 | 0.6401 | 14.70 |
| small-b-sampled | test | 0.2973 | 0.2925 | 0.2921 | 0.5744 | 0.5124 | 0.6177 | 0.6097 | 15.23 |
| medium-b-mixed | train | 1.7488 | 0.7531 | 0.7414 | 0.8354 | 0.8078 | 1.0896 | 1.0931 | 28.31 |
| medium-b-sampled | valid | 0.3383 | 0.3351 | 0.3334 | 0.6190 | 0.5522 | 0.6888 | 0.6893 | 31.31 |
| medium-b-sampled | test | 0.3451 | 0.3420 | 0.3409 | 0.6515 | 0.5847 | 0.7057 | 0.7193 | 30.98 |
| large-b-mixed | train | 1.8088 | 0.7873 | 0.7743 | 0.7228 | 0.6475 | 0.9583 | 0.8844 | 32.15 |
| large-b-sampled | valid | 1.6622 | 0.7255 | 0.7180 | 0.8365 | 0.7381 | 1.1129 | 1.0693 | 34.16 |
| large-b-sampled | test | 1.8280 | 0.7834 | 0.7726 | 0.8752 | 0.7606 | 0.9178 | 0.8269 | 37.25 |
| huge-b-mixed | train | 0.9525 | 0.5603 | 0.5533 | 1.0181 | 0.9322 | 1.0459 | 1.0063 | 38.64 |
| huge-b-sampled | valid | 0.3925 | 0.3893 | 0.3869 | 0.8705 | 0.7455 | 0.9367 | 0.8290 | 55.55 |
| huge-b-sampled | test | 0.3840 | 0.3814 | 0.3788 | 0.7908 | 0.6683 | 0.8143 | 0.7651 | 50.33 |

Interpretation:

- P14-all improves FFX/σ almost everywhere but still trails INLA on hard mixed rows.
- P14-deep improves over P14-all on every medium/large/huge row and remains below
  `~56 ms/dataset` CPU in this benchmark.
- P14-deep has localized BLUP regressions on small-mixed and medium-sampled rows. BLUP is
  not the limiting metric for NPE context, but do not make P14-deep the default until the
  full required-suite rerun confirms the tradeoff is acceptable.
- The β-jump BLUP fallback is rare but useful. On first-512 large-b-sampled/test it changed
  P14 BLUP NRMSE from `0.983` to `0.901` while keeping FFX `0.911` and σ `0.520`.

External References
-------------------

**CAVI:** P14-deep beats the stored CAVI table on FFX for all 12 CAVI-subset rows and on
σ/BLUP for nearly all rows. Remaining caveats:

- medium-b-sampled/test BLUP: P14-deep `0.966` vs CAVI `0.831`
- huge-b-sampled/test σ: P14-deep `0.796` vs CAVI `0.784`

CAVI is no longer the primary target.

**INLA:** INLA remains the useful accuracy target, especially mixed/train FFX:

| Dataset | part | P14-deep FFX | INLA FFX | P14-deep σ | INLA σ | P14-deep BLUP | INLA BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2687 | 0.451 | 0.5170 | 0.567 | 0.6198 | 0.618 |
| small-b-sampled | test | 0.2921 | 0.447 | 0.5124 | 0.556 | 0.6097 | 0.625 |
| medium-b-mixed | train | 0.7414 | 0.331 | 0.8078 | 0.519 | 1.0931 | 0.648 |
| medium-b-sampled | test | 0.3409 | 0.400 | 0.5847 | 4.490 † | 0.7193 | 0.692 |
| large-b-mixed | train | 0.7743 | 0.323 | 0.6475 | 0.521 | 0.8844 | 0.676 |
| large-b-sampled | test | 0.7726 | 0.365 | 0.7606 | 0.603 | 0.8269 | 0.710 |
| huge-b-mixed | train | 0.5533 | 0.330 | 0.9322 | 0.550 | 1.0063 | 0.713 |
| huge-b-sampled | test | 0.3788 | 0.394 | 0.6683 | 0.579 | 0.7651 | 0.740 |

† INLA medium-b-sampled σ outlier from stored reference table.

Next Steps
----------

1. **Tune P14-deep schedule before changing architecture.**
   Run a focused schedule grid on the hard INLA-gap rows:

   - current deep: `20/4/8`, `lr=0.05`
   - `24/4/8`, `lr=0.05`
   - `24/4/10`, `lr=0.05`
   - `20/4/8`, `lr=0.035`

   Primary rows: `medium-b-mixed/train`, `large-b-mixed/train`,
   `large-b-sampled/test`, `huge-b-mixed/train`.

2. **Analyze accepted vs rejected P14 cases.**
   P14-deep acceptance is often `~0.8-0.9`, not 1.0. Quantify whether rejected datasets
   still have high current error. If yes, the objective acceptance gate may be too
   conservative or misaligned with the summaries the NPE needs.

3. **Diagnose remaining hard mixed-row FFX gap.**
   For high-FFX-error datasets, compare current vs P14:

   - β error and β jump
   - target/base-target delta
   - σ movement
   - active `d/q/m/n`
   - whether acceptance rejected the P14 update

   The remaining gap is likely β-mode quality, not BLUP postprocessing.

4. **Try a cheap β-only post-pass on P14 σ.**
   After P14-deep settles σ, run a few nested β Newton updates with σ fixed, then refresh
   BLUPs. This is lower risk than adding new approximation machinery and directly targets
   under-converged β.

5. **Only if schedule/post-pass saturates: test a tiny multi-start for high-risk rows.**
   Candidate starts:

   - current stats β/σ
   - current stats β with a more aggressive tiny-σ continuation

   Pick by Laplace target. Keep this gated to hard/high-risk rows only.

Deprioritized
-------------

- **Full PyTorch INLA:** too many hyperparameter-mode solves for the runtime target.
- **R-INLA backend:** not suitable for batched processing/GPU utilization.
- **More PQL-local patches:** low expected upside after P12/P13.
- **EP or Pólya-Gamma variational GLMM:** possible future branch, but more moving parts than
  the current Laplace-EB path.

Compact Archive
---------------

- **P11/INLA-lite grid σ integration:** improved σ but catastrophically worsened FFX because
  β averaging pulled estimates toward the low-σ/OLS regime. Reverted.
- **P12 nested β/b̂_g Newton:** current default β refinement. Useful, but cannot fully escape
  poor PQL basins at high d.
- **P13 cold starts:** resetting β and/or b̂_g away from PQL caused FE/RE confounding. With
  weak RE shrinkage, b̂_g absorbs fixed-effect signal before β can learn.
- **P13 outer nAGQ loop:** too slow and unstable because nAGQ σ updates and P12 M-step optimize
  different objectives.

Useful Commands
---------------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py \
    --family b --sizes medium large huge --methods current p14_all p14_deep \
    --batch-size 32 --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
