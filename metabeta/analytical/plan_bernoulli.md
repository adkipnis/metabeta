Bernoulli GLMM Plan
===================

Last updated: 2026-05-16

Goal
----

Provide fast, high-accuracy Bernoulli GLMM summaries for the hierarchical NPE context.
The analytical estimator should be stable, prior-aware, and cheap; it is not intended to
be a full posterior engine.

Do not add an amortized correction branch here. `glmm()` summaries are already consumed by
the hierarchical NPE, which is the correction mechanism.

Do not pursue R-INLA as a backend or full PyTorch INLA as the main path. Use INLA concepts
only where they stay compatible with batched CPU/GPU processing and the `~100 ms/dataset`
target.

Current Candidate
-----------------

**P14-cal** is the best Bernoulli analytical candidate so far:

- Laplace-EB refinement over β and diagonal log σ with nested per-group random-effect modes.
- Objective acceptance gate against the incoming PQL/IRLS path.
- Separation guard: output β is capped at `±3` only when max active `|β| > 8`.
- Sigma calibration: cap σ at `2.0 * tau_rfx` for effective `d >= 5`, then recompute BLUP
  modes with the calibrated σ.

The candidate is exposed through kwargs, not defaults:

```python
glmm(
    ...,
    bernoulli_laplace_eb=True,
    bernoulli_laplace_eb_steps=24,
    bernoulli_laplace_eb_inner=4,
    bernoulli_laplace_eb_final=8,
    bernoulli_laplace_eb_lr=0.05,
    bernoulli_laplace_eb_beta_output_cap=3.0,
    bernoulli_laplace_eb_beta_output_cap_trigger=8.0,
    bernoulli_laplace_eb_sigma_prior_cap=2.0,
    bernoulli_laplace_eb_sigma_prior_cap_min_d=5,
)
```

Performance Snapshot
--------------------

Matched first-1000 benchmark, CPU ms/dataset. Lower NRMSE is better.

| Dataset | part | FFX | σ | BLUP | INLA FFX | INLA σ | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2683 | 0.5119 | 0.6127 | 0.451 | 0.567 | 0.618 | 18.06 |
| small-b-sampled | test | 0.2925 | 0.5041 | 0.6089 | 0.447 | 0.556 | 0.625 | 16.95 |
| medium-b-mixed | train | 0.3127 | 0.5426 | 0.6719 | 0.331 | 0.519 | 0.648 | 35.48 |
| medium-b-sampled | test | 0.3393 | 0.6007 | 0.7103 | 0.400 | 4.490 † | 0.692 | 38.96 |
| large-b-mixed | train | 0.3316 | 0.5529 | 0.6868 | 0.323 | 0.521 | 0.676 | 40.53 |
| large-b-sampled | test | 0.3574 | 0.6229 | 0.7283 | 0.365 | 0.603 | 0.710 | 47.58 |
| huge-b-mixed | train | 0.3352 | 0.6033 | 0.7329 | 0.330 | 0.550 | 0.713 | 49.66 |
| huge-b-sampled | test | 0.3778 | 0.6354 | 0.7549 | 0.394 | 0.579 | 0.740 | 61.79 |

† Stored INLA medium-b-sampled σ is an outlier and should not drive decisions.

Takeaways
---------

- FFX is effectively closed relative to INLA.
- Remaining INLA gaps are mostly σ/BLUP: about `0.02-0.06` σ NRMSE and
  `0.01-0.025` BLUP NRMSE on medium/large/huge rows.
- P14-cal closes the previous mixed BLUP failure: medium `1.1051 -> 0.6719`,
  large `0.8719 -> 0.6868`, huge `1.0147 -> 0.7329`.
- The sigma cap fires on about `6-9%` of medium/large/huge rows and zero small rows.
- There are small sampled-row tradeoffs, especially around medium/huge sampled BLUP.

Next Steps
----------

1. **Run the downstream NPE-context smoke/ablation for P14-cal.**
   Promote the candidate only if the consumer model likes the calibrated summaries.

2. **Sweep only the sigma cap if more analytical accuracy is needed.**
   Test `sigma_prior_cap in {1.75, 2.0, 2.25, 2.5}` with `min_d=5`. This is the
   most likely low-risk way to reduce the residual σ/BLUP gap without adding machinery.

3. **Consider one conditional sampled-row rule only if the sweep exposes a tradeoff.**
   Candidate rule: apply sigma calibration only when P14 σ is far above `tau_rfx` or when
   existing instability diagnostics fire. Keep this as one predicate, not a new optimizer.

4. **Defer β optimizer work, multi-starts, EP, and full INLA.**
   They add moving parts while the remaining gap is no longer β-dominated.

Commands
--------

```bash
uv run python experiments/analytical/glmm_required_benchmark.py \
    --family b --sizes small medium large huge --methods p14_cal \
    --batch-size 32 --max-datasets 1000

uv run python experiments/analytical/glmm_required_benchmark.py \
    --family b --methods p14_cal \
    --combos medium-b-mixed:train:2 large-b-mixed:train:2 \
        large-b-sampled:test huge-b-mixed:train:2 \
    --batch-size 32 --max-datasets 1000 --cal-sigma-prior-cap 2.5

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```

Retired Lines
-------------

- R-INLA backend and full PyTorch INLA: incompatible with the throughput target.
- P11 σ grid integration: worsened FFX by pulling β toward low-σ/OLS regimes.
- P13 cold starts and nAGQ outer loops: unstable FE/RE confounding or too slow.
- Extra β-only Newton and multi-starts: lower expected value after the β output guard.
