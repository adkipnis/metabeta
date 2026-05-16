Bernoulli GLMM Plan
===================

Last updated: 2026-05-16 (after P14 sigma/BLUP calibration pass)

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
- `bernoulli_laplace_eb_sigma_prior_cap` default `None`
- `bernoulli_laplace_eb_sigma_prior_cap_min_d` default `None`
- `bernoulli_laplace_eb_recompute_blup_after_calibration` default `True`

P14 diagnostics:

- `laplace_eb_accept`
- `laplace_eb_steps`
- `laplace_eb_target`
- `laplace_eb_base_target`
- `laplace_eb_blup_fallback`
- `laplace_eb_beta_jump`
- `laplace_eb_beta_output_capped`
- `laplace_eb_sigma_prior_capped`

**P15 auto gate:** currently uses effective `d >= 4`, mean estimated `σ_rfx >= 0.75`, or max
fitted `|η| >= 8`. These are configurable through `bernoulli_laplace_eb_gate_*` kwargs. In
matched medium+ benchmarks, this gate selects 100% of datasets, so it is a small-scale cost
saver rather than a medium/large accuracy selector.

Current Performance
-------------------

Matched first-1000 benchmark, 2026-05-16. `P14-guard` means 24 outer steps, 4 inner mode
steps, 8 final mode steps, `lr=0.05`, and a separation-only β summary guard:
`beta_output_cap=3`, `beta_output_cap_trigger=8`. `P14-cal` adds
`sigma_prior_cap=2.0`, `sigma_prior_cap_min_d=5`, and BLUP recomputation after sigma
calibration. CPU timings are per dataset.

| Dataset | part | P14-guard FFX | P14-cal FFX | P14-guard σ | P14-cal σ | P14-guard BLUP | P14-cal BLUP | cal ms/ds | σ cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2683 | 0.2683 | 0.5119 | 0.5119 | 0.6127 | 0.6127 | 18.06 | 0.000 |
| small-b-sampled | valid | 0.3127 | 0.3127 | 0.5385 | 0.5385 | 0.6386 | 0.6386 | 16.14 | 0.000 |
| small-b-sampled | test | 0.2925 | 0.2925 | 0.5041 | 0.5041 | 0.6089 | 0.6089 | 16.95 | 0.000 |
| medium-b-mixed | train | 0.3127 | 0.3127 | 0.8186 | 0.5426 | 1.1051 | 0.6719 | 35.48 | 0.067 |
| medium-b-sampled | valid | 0.3328 | 0.3328 | 0.5469 | 0.5736 | 0.6874 | 0.6944 | 39.50 | 0.062 |
| medium-b-sampled | test | 0.3393 | 0.3393 | 0.5789 | 0.6007 | 0.7173 | 0.7103 | 38.96 | 0.060 |
| large-b-mixed | train | 0.3316 | 0.3316 | 0.6830 | 0.5529 | 0.8719 | 0.6868 | 40.53 | 0.067 |
| large-b-sampled | valid | 0.3672 | 0.3672 | 0.7330 | 0.6045 | 1.0670 | 0.7422 | 43.76 | 0.062 |
| large-b-sampled | test | 0.3574 | 0.3574 | 0.7431 | 0.6229 | 0.8331 | 0.7283 | 47.58 | 0.063 |
| huge-b-mixed | train | 0.3352 | 0.3352 | 0.9104 | 0.6033 | 1.0147 | 0.7329 | 49.66 | 0.093 |
| huge-b-sampled | valid | 0.3857 | 0.3857 | 0.7260 | 0.6149 | 0.8019 | 0.7662 | 68.50 | 0.058 |
| huge-b-sampled | test | 0.3778 | 0.3778 | 0.6455 | 0.6354 | 0.7523 | 0.7549 | 61.79 | 0.067 |

Interpretation:

- Plain P14-deep improves over P14-all on every medium/large/huge row but still trails INLA
  on hard mixed FFX because rare separation-scale β summaries dominate aggregate NRMSE.
- P14-guard fixes that specific FFX failure mode with a very small trigger rate: about
  `0.1-0.2%` on hard rows and zero on most easy rows. It changes only `beta_est`; σ and BLUP
  are inherited from the 24/4/8 P14 solve.
- P14-cal leaves β unchanged and closes most of the mixed-row σ/BLUP gap with a small,
  prior-scale cap on `6-9%` of medium/large/huge rows. The `d >= 5` guard leaves small rows
  unchanged.
- P14-cal has small sampled-row tradeoffs: medium/huge sampled BLUP can move by about
  `0.003-0.007` in either direction, while large sampled improves substantially.
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

| Dataset | part | P14-cal FFX | INLA FFX | P14-cal σ | INLA σ | P14-cal BLUP | INLA BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2683 | 0.451 | 0.5119 | 0.567 | 0.6127 | 0.618 |
| small-b-sampled | test | 0.2925 | 0.447 | 0.5041 | 0.556 | 0.6089 | 0.625 |
| medium-b-mixed | train | 0.3127 | 0.331 | 0.5426 | 0.519 | 0.6719 | 0.648 |
| medium-b-sampled | test | 0.3393 | 0.400 | 0.6007 | 4.490 † | 0.7103 | 0.692 |
| large-b-mixed | train | 0.3316 | 0.323 | 0.5529 | 0.521 | 0.6868 | 0.676 |
| large-b-sampled | test | 0.3574 | 0.365 | 0.6229 | 0.603 | 0.7283 | 0.710 |
| huge-b-mixed | train | 0.3352 | 0.330 | 0.6033 | 0.550 | 0.7329 | 0.713 |
| huge-b-sampled | test | 0.3778 | 0.394 | 0.6354 | 0.579 | 0.7549 | 0.740 |

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

Sigma calibration check:

- The mixed-row σ/BLUP error is broad rather than limited to the rare β-cap rows.
- Capping σ at `2*tau_rfx` and recomputing BLUP modes improved first-1000 mixed BLUP NRMSE:
  medium `1.1051 -> 0.6719`, large `0.8719 -> 0.6868`, huge `1.0147 -> 0.7329`.
- A `d >= 5` guard prevents this branch from touching small rows, where P14-guard already
  beats or matches the references.

Next Steps
----------

1. **Run the downstream NPE-context smoke/ablation for P14-cal.**
   P14-cal is exposed by kwargs and benchmark CLI only. It is the strongest analytical
   candidate so far, but defaults should wait for the consumer model check because sampled-row
   BLUP tradeoffs are small but real.

2. **If the downstream check dislikes P14-cal, try one simpler variant before stopping.**
   Use `sigma_prior_cap=2.5` with the same `d >= 5` guard. Diagnostics suggest slightly weaker
   BLUP gains but lower risk of over-shrinking sampled rows.

3. **Defer β-only Newton and multi-start.**
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
    --p14-beta-output-cap 3 --p14-beta-output-cap-trigger 8 \
    --p14-sigma-prior-cap 2.0 --p14-sigma-prior-cap-min-d 5

uv run python experiments/analytical/glmm_required_benchmark.py \
    --family b --methods p14_custom \
    --combos medium-b-mixed:train:2 large-b-mixed:train:2 \
        large-b-sampled:test huge-b-mixed:train:2 \
    --batch-size 32 --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```
