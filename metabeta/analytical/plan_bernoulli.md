Bernoulli GLMM Plan
===================

Last updated: 2026-05-15 (after P14 schedule/diagnostic pass)

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
- `bernoulli_laplace_eb_beta_output_cap` default `None`
- `bernoulli_laplace_eb_beta_output_cap_trigger` default `None`

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

Matched first-1000 benchmark, 2026-05-15. `p14_guard` means 24 outer steps, 4 inner mode
steps, 8 final mode steps, `lr=0.05`, and a separation-only β summary guard:
`beta_output_cap=3`, `beta_output_cap_trigger=8`. CPU timings are per dataset.

| Dataset | part | Current FFX | P14-deep FFX | P14-guard FFX | Current σ | P14-guard σ | Current BLUP | P14-guard BLUP | guard ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2710 | 0.2687 | 0.2683 | 0.5712 | 0.5119 | 0.6176 | 0.6127 | 19.00 |
| small-b-sampled | valid | 0.3186 | 0.3133 | 0.3127 | 0.6041 | 0.5385 | 0.6502 | 0.6386 | 14.35 |
| small-b-sampled | test | 0.2973 | 0.2921 | 0.2925 | 0.5744 | 0.5041 | 0.6177 | 0.6089 | 14.95 |
| medium-b-mixed | train | 1.7488 | 0.7414 | 0.3127 | 0.8354 | 0.8186 | 1.0896 | 1.1051 | 30.12 |
| medium-b-sampled | valid | 0.3383 | 0.3334 | 0.3328 | 0.6190 | 0.5469 | 0.6888 | 0.6874 | 33.10 |
| medium-b-sampled | test | 0.3451 | 0.3409 | 0.3393 | 0.6515 | 0.5789 | 0.7057 | 0.7173 | 32.53 |
| large-b-mixed | train | 1.8088 | 0.7743 | 0.3316 | 0.7228 | 0.6830 | 0.9583 | 0.8719 | 33.56 |
| large-b-sampled | valid | 1.6622 | 0.7180 | 0.3672 | 0.8365 | 0.7330 | 1.1129 | 1.0670 | 35.95 |
| large-b-sampled | test | 1.8280 | 0.7726 | 0.3574 | 0.8752 | 0.7431 | 0.9178 | 0.8331 | 40.52 |
| huge-b-mixed | train | 0.9525 | 0.5533 | 0.3352 | 1.0181 | 0.9104 | 1.0459 | 1.0147 | 41.58 |
| huge-b-sampled | valid | 0.3925 | 0.3869 | 0.3857 | 0.8705 | 0.7260 | 0.9367 | 0.8019 | 58.86 |
| huge-b-sampled | test | 0.3840 | 0.3788 | 0.3778 | 0.7908 | 0.6455 | 0.8143 | 0.7523 | 53.10 |

Interpretation:

- Plain P14-deep improves over P14-all on every medium/large/huge row but still trails INLA
  on hard mixed FFX because rare separation-scale β summaries dominate aggregate NRMSE.
- P14-guard fixes that specific FFX failure mode with a very small trigger rate: about
  `0.1-0.2%` on hard rows and zero on most easy rows. It changes only `beta_est`; σ and BLUP
  are inherited from the 24/4/8 P14 solve.
- P14-guard is now close to INLA on hard mixed FFX, but σ/BLUP remain behind INLA on
  medium/large/huge mixed rows.
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

| Dataset | part | P14-guard FFX | INLA FFX | P14-guard σ | INLA σ | P14-guard BLUP | INLA BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2683 | 0.451 | 0.5119 | 0.567 | 0.6127 | 0.618 |
| small-b-sampled | test | 0.2925 | 0.447 | 0.5041 | 0.556 | 0.6089 | 0.625 |
| medium-b-mixed | train | 0.3127 | 0.331 | 0.8186 | 0.519 | 1.1051 | 0.648 |
| medium-b-sampled | test | 0.3393 | 0.400 | 0.5789 | 4.490 † | 0.7173 | 0.692 |
| large-b-mixed | train | 0.3316 | 0.323 | 0.6830 | 0.521 | 0.8719 | 0.676 |
| large-b-sampled | test | 0.3574 | 0.365 | 0.7431 | 0.603 | 0.8331 | 0.710 |
| huge-b-mixed | train | 0.3352 | 0.330 | 0.9104 | 0.550 | 1.0147 | 0.713 |
| huge-b-sampled | test | 0.3778 | 0.394 | 0.6455 | 0.579 | 0.7523 | 0.740 |

† INLA medium-b-sampled σ outlier from stored reference table.

Diagnostic Results
------------------

Focused schedule grid on hard rows, first 1000 datasets:

| Schedule | medium mixed FFX | large mixed FFX | large sampled/test FFX | huge mixed FFX |
| --- | ---: | ---: | ---: | ---: |
| `20/4/8`, `lr=0.05` | 0.7414 | 0.7743 | 0.7726 | 0.5533 |
| `24/4/8`, `lr=0.05` | 0.7359 | 0.7676 | 0.7662 | 0.5502 |
| `24/4/10`, `lr=0.05` | 0.7359 | 0.7676 | 0.7662 | 0.5502 |
| `20/4/8`, `lr=0.035` | 0.7503 | 0.7841 | 0.7806 | 0.5580 |
| `24/4/8` + β guard | 0.3127 | 0.3316 | 0.3574 | 0.3352 |

Acceptance-gate check:

- Accepted rows carry the hard FFX error; rejected rows are already easy. On the four hard
  rows, accepted FFX was roughly `0.58-0.81` after plain P14, while rejected rows were
  `0.24-0.29` and unchanged by construction.
- Top residual cases are separation-scale β summaries: current β hits `±50`, plain P14 hits
  its `±20` clamp, but the true β values are ordinary in several cases. These cases are rare
  but dominate aggregate FFX NRMSE.

Next Steps
----------

1. **Make the P14-guard candidate first-class if downstream checks agree.**
   It is currently exposed by kwargs and benchmark CLI only. Before changing defaults, run
   the downstream NPE-context smoke/ablation that consumes `glmm()` summaries. The guard is
   deliberately simple: it only protects `beta_est` from rare separation-scale summaries.

2. **Close the σ/BLUP gap on mixed rows.**
   P14-guard solved the main FFX gap, but mixed-row σ/BLUP still trail INLA. The next
   low-moving-parts branch is a σ shrinkage diagnostic, not a new posterior engine:

   - compare P14 σ to prior scale `tau_rfx` and true σ on accepted hard mixed rows
   - test a guarded convex blend between P14 σ and prior-scale σ only when P14 σ is far
     above the prior scale and β was separation-guarded
   - recompute BLUP modes after any σ adjustment

3. **Check consistency of β guard with BLUP recomputation.**
   Current P14-guard changes only `beta_est`; BLUPs still come from the uncapped P14 mode.
   This is acceptable for NPE context if β and BLUP are separate summary channels, but the
   next cheap check is to recompute BLUP modes with guarded β and fixed σ on triggered rows.

4. **Defer β-only Newton and multi-start.**
   The diagnostics showed the rejected P14 cases are low-error: accepted rows had high current
   FFX and rejected rows had current/P14 FFX around `0.24-0.29`. More optimizer work is now
   lower priority than σ/BLUP calibration.

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
    --family b --sizes small medium large huge --methods p14_custom \
    --batch-size 32 --max-datasets 1000 \
    --p14-steps 24 --p14-inner 4 --p14-final 8 --p14-lr 0.05 \
    --p14-beta-output-cap 3 --p14-beta-output-cap-trigger 8

uv run python experiments/analytical/glmm_required_benchmark.py \
    --family b --methods p14_custom \
    --combos medium-b-mixed:train:2 large-b-mixed:train:2 \
        large-b-sampled:test huge-b-mixed:train:2 \
    --batch-size 32 --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
