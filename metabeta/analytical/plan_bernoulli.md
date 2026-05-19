Bernoulli GLMM Plan
===================

Last updated: 2026-05-19

Goal
----

Provide a fast, high-accuracy analytical Bernoulli GLMM estimator. The estimator should be
stable, prior-aware, and cheap; it is not intended to be a full posterior engine.

Do not pursue R-INLA as a backend or full PyTorch INLA as the main path. Use INLA concepts
only where they stay compatible with batched CPU/GPU processing and the `~100 ms/dataset`
target.

Default Path
------------

**Bernoulli EB** is now the default Bernoulli analytical path in `glmm()`:

- Laplace-EB refinement over β and diagonal log σ with nested per-group random-effect modes.
- Objective acceptance gate against the incoming PQL/IRLS path.
- Separation guard: output β is capped at `±3` only when max active `|β| > 8`.
- Sigma calibration: cap σ at `2.5 * tau_rfx` for effective `d >= 5`, then recompute BLUP
  modes with the calibrated σ.

The explicit preset remains available for benchmarks and diagnostics:

```python
glmm(
    ...,
    bernoulli_laplace_eb='bernoulli_eb',
)
```

The old PQL/IRLS path remains available with `bernoulli_laplace_eb=False`.
Individual preset values can still be overridden by explicit kwargs for diagnostics.

Performance Snapshot
--------------------

Matched first-1000 per comparison row, 12k matched datasets total. CPU ms/dataset.
Lower NRMSE is better.

| Dataset | part | Legacy FFX | EB FFX | Legacy σ | EB σ | Legacy BLUP | EB BLUP | INLA FFX | INLA σ | INLA BLUP | ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b-mixed | train | 0.2710 | 0.2683 | 0.5712 | 0.5119 | 0.6176 | 0.6127 | 0.451 | 0.567 | 0.618 | 17.20 |
| small-b-sampled | valid | 1.0642 | 0.3127 | 0.6868 | 0.5385 | 0.6853 | 0.6386 | 0.559 | 0.539 | 0.639 | 18.71 |
| small-b-sampled | test | 0.9899 | 0.2925 | 0.6788 | 0.5041 | 0.6600 | 0.6089 | 0.446 | 0.517 | 0.618 | 19.65 |
| medium-b-mixed | train | 1.7488 | 0.3127 | 0.8354 | 0.5389 | 1.0896 | 0.6865 | 0.332 | 0.522 | 0.648 | 35.21 |
| medium-b-sampled | valid | 2.5612 | 0.3328 | 0.7529 | 0.5572 | 0.8523 | 0.6905 | 0.345 | 0.541 | 0.670 | 49.70 |
| medium-b-sampled | test | 2.6535 | 0.3393 | 0.8768 | 0.5844 | 1.0345 | 0.7069 | 0.400 | 0.563 | 0.690 | 48.30 |
| large-b-mixed | train | 1.8088 | 0.3316 | 0.7228 | 0.5423 | 0.9583 | 0.6853 | 0.323 | 0.521 | 0.676 | 39.76 |
| large-b-sampled | valid | 3.6972 | 0.3672 | 1.1333 | 0.5941 | 1.6514 | 0.7418 | 0.371 | 0.591 | 0.737 | 53.15 |
| large-b-sampled | test | 3.6324 | 0.3574 | 1.0200 | 0.6196 | 1.2414 | 0.7271 | 0.365 | 0.569 | 0.708 | 50.33 |
| huge-b-mixed | train | 0.9525 | 0.3352 | 1.0181 | 0.5998 | 1.0459 | 0.7367 | 0.330 | 0.550 | 0.713 | 49.03 |
| huge-b-sampled | valid | 5.5916 | 0.3857 | 1.3181 | 0.6111 | 1.9261 | 0.7667 | 0.393 | 0.569 | 0.751 | 71.17 |
| huge-b-sampled | test | 4.3353 | 0.3778 | 1.1359 | 0.6265 | 1.5815 | 0.7528 | 0.394 | 0.548 | 0.737 | 64.41 |

Sampled valid/test INLA cells were refreshed on 2026-05-19 with explicit diagonal
R-INLA; logs are under `experiments/analytical/inla_runs/bernoulli_sampled/`.

Takeaways
---------

- Bernoulli EB improves over the legacy analytical path on every comparison row for FFX and σ.
- Bernoulli EB improves BLUP over the refreshed legacy baseline on every sampled row and
  closes the previous mixed-row BLUP failures.
- FFX is effectively closed relative to INLA and is better than INLA on all sampled rows.
- Remaining INLA gaps are mostly σ/BLUP: about `0.02-0.08` σ NRMSE and
  `0.005-0.025` BLUP NRMSE on medium/large/huge sampled rows.
- Bernoulli EB closes the previous mixed BLUP failure: medium `1.1051 -> 0.6865`,
  large `0.8719 -> 0.6853`, huge `1.0147 -> 0.7367`.
- The sigma cap fires on about `2-4%` of medium/large/huge rows and zero small rows.
- The cap sweep favored `2.5` as the best balanced setting: `1.75-2.0` slightly helps
  medium/huge mixed BLUP but over-shrinks more sampled rows; `2.5` is better on σ and most
  sampled-row BLUPs.
- A conditional cap requiring β/BLUP instability was benchmarked and rejected. It reduced cap
  fires but worsened medium/huge mixed σ and BLUP enough to fail the full-table criterion.

Next Steps
----------

1. **Freeze the analytical Bernoulli estimator around the default Bernoulli EB path.**
   Future changes should be regression fixes or clearly benchmarked simplifications.

2. **Use the full mixed plus sampled benchmark as the promotion gate.**
   Small local improvements are not enough if they move σ/BLUP backward on mixed or sampled rows.

3. **Do not add more optimizer machinery for the current residual gap.**
   The remaining analytical improvements are likely small and conditional. β optimizer work,
   multi-starts, EP, and full INLA are deferred.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family b --methods raw current bernoulli_eb \
    --combos small-b-mixed:train:2 small-b-sampled:valid small-b-sampled:test \
        medium-b-mixed:train:2 medium-b-sampled:valid medium-b-sampled:test \
        large-b-mixed:train:2 large-b-sampled:valid large-b-sampled:test \
        huge-b-mixed:train:2 huge-b-sampled:valid huge-b-sampled:test \
    --batch-size 32 --max-datasets 1000

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids small-b-sampled --partition valid \
    --n-inla 1000 --n-total 1000 --analytical-methods raw,current \
    --re-correlation diagonal
# Repeat the INLA command for each sampled size and valid/test partition.

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/glmm.py metabeta/analytical/map.py \
    experiments/analytical/glmm_required_benchmark.py tests/utils/test_glmm.py
```

Retired Lines
-------------

- R-INLA backend and full PyTorch INLA: incompatible with the throughput target.
- σ grid integration: worsened FFX by pulling β toward low-σ/OLS regimes.
- Cold starts and nAGQ outer loops: unstable FE/RE confounding or too slow.
- Extra β-only Newton and multi-starts: lower expected value after the β output guard.
- Conditional sigma cap requiring β/BLUP instability: reduced cap fires but worsened the
  full 8k benchmark, especially medium/huge mixed rows.
