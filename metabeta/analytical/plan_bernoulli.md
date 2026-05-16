Bernoulli GLMM Plan
===================

Last updated: 2026-05-16

Goal
----

Provide a fast, high-accuracy analytical Bernoulli GLMM estimator. The estimator should be
stable, prior-aware, and cheap; it is not intended to be a full posterior engine.

Do not pursue R-INLA as a backend or full PyTorch INLA as the main path. Use INLA concepts
only where they stay compatible with batched CPU/GPU processing and the `~100 ms/dataset`
target.

Current Candidate
-----------------

**P14-cal** is the best Bernoulli analytical candidate so far:

- Laplace-EB refinement over β and diagonal log σ with nested per-group random-effect modes.
- Objective acceptance gate against the incoming PQL/IRLS path.
- Separation guard: output β is capped at `±3` only when max active `|β| > 8`.
- Sigma calibration: cap σ at `2.5 * tau_rfx` for effective `d >= 5`, then recompute BLUP
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
    bernoulli_laplace_eb_sigma_prior_cap=2.5,
    bernoulli_laplace_eb_sigma_prior_cap_min_d=5,
)
```

Performance Snapshot
--------------------

Matched first-1000 benchmark, CPU ms/dataset. Lower NRMSE is better.

| Dataset | part | FFX | σ | BLUP | INLA FFX | INLA σ | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2683 | 0.5119 | 0.6127 | 0.451 | 0.567 | 0.618 | 17.54 |
| small-b-sampled | test | 0.2925 | 0.5041 | 0.6089 | 0.447 | 0.556 | 0.625 | 17.36 |
| medium-b-mixed | train | 0.3127 | 0.5389 | 0.6865 | 0.331 | 0.519 | 0.648 | 35.58 |
| medium-b-sampled | test | 0.3393 | 0.5844 | 0.7069 | 0.400 | 4.490 † | 0.692 | 38.46 |
| large-b-mixed | train | 0.3316 | 0.5423 | 0.6853 | 0.323 | 0.521 | 0.676 | 40.44 |
| large-b-sampled | test | 0.3574 | 0.6196 | 0.7271 | 0.365 | 0.603 | 0.710 | 48.74 |
| huge-b-mixed | train | 0.3352 | 0.5998 | 0.7367 | 0.330 | 0.550 | 0.713 | 50.29 |
| huge-b-sampled | test | 0.3778 | 0.6265 | 0.7528 | 0.394 | 0.579 | 0.740 | 65.43 |

† Stored INLA medium-b-sampled σ is an outlier and should not drive decisions.

Takeaways
---------

- FFX is effectively closed relative to INLA.
- Remaining INLA gaps are mostly σ/BLUP: about `0.02-0.05` σ NRMSE and
  `0.01-0.025` BLUP NRMSE on medium/large/huge rows.
- P14-cal closes the previous mixed BLUP failure: medium `1.1051 -> 0.6865`,
  large `0.8719 -> 0.6853`, huge `1.0147 -> 0.7367`.
- The sigma cap fires on about `2-4%` of medium/large/huge rows and zero small rows.
- The cap sweep favored `2.5` as the best balanced setting: `1.75-2.0` slightly helps
  medium/huge mixed BLUP but over-shrinks more sampled rows; `2.5` is better on σ and most
  sampled-row BLUPs.

Next Steps
----------

1. **Promote P14-cal to a first-class analytical preset.**
   Keep the current default path unchanged until the final benchmark pass is accepted, but
   expose the retained schedule/guards as a named preset or compact option instead of a long
   list of kwargs.

2. **Do not add more optimizer machinery for the current residual gap.**
   The remaining analytical improvements are likely small and conditional. Keep the candidate
   simple unless benchmark diagnostics identify a real failure mode.

3. **Consider one conditional sampled-row rule only if benchmarks justify it.**
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
    --batch-size 32 --max-datasets 1000 --cal-sigma-prior-cap 2.0

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical experiments/analytical
```

Retired Lines
-------------

- R-INLA backend and full PyTorch INLA: incompatible with the throughput target.
- P11 σ grid integration: worsened FFX by pulling β toward low-σ/OLS regimes.
- P13 cold starts and nAGQ outer loops: unstable FE/RE confounding or too slow.
- Extra β-only Newton and multi-starts: lower expected value after the β output guard.
