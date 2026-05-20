Poisson GLMM Plan
=================

Last updated: 2026-05-20 (fresh small/medium/large 1k benchmark)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA is a
reference implementation only; R-INLA-as-backend and full PyTorch INLA remain out of
scope. The next retained path should be simple enough to trust and useful as context for
downstream models.

Current Working Model
---------------------

Default Poisson path:

1. RAW/PQL Poisson GLMM initialization.
2. Diagonal single-mode Laplace EB over β and diagonal σ_rfx.
3. Per-row accept gate against RAW/PQL.
4. RAW/PQL BLUP fallback, because direct Poisson EB BLUP modes regressed most cells.
5. Accepted-row σ cap at `2.5 * tau_rfx` for effective `d >= 5`.
6. Marginal-mean β correction for `d >= 5`.
7. Targeted σ-offset grid for `d >= 5` and `q <= 2`.

The σ-offset grid is β-only: it tests a few σ scales to generate alternative marginal-mean
β candidates, accepts by a cheap AGQ-style marginal posterior, then writes back β while
keeping EB σ and RAW/PQL BLUP unchanged.

Best current prototype:

```python
glmm(
    ...,
    poisson_marginal_beta_min_d=1,
    poisson_sigma_grid_min_d=1,
)
```

This relaxes the two `d >= 5` gates. It substantially improves small rows and leaves
medium/large unchanged because those gates already fired.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU run on 2026-05-20. Lower NRMSE is better.
INLA values are the current first-1000 diagonal R-INLA references.

| Dataset | part | RAW FFX | default FFX | best proto FFX | INLA FFX | best σ | INLA σ | best BLUP | INLA BLUP | best ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.4806 | 0.4269 | 0.2839 | 0.1835 | 0.5272 | 0.3404 | 0.5713 | 0.4936 | 53.3 |
| small-p-sampled | valid | 0.7351 | 0.6375 | 0.3277 | 0.2276 | 0.5645 | 0.4356 | 0.5823 | 0.5309 | 46.4 |
| small-p-sampled | test | 0.6525 | 0.5429 | 0.2982 | 0.1997 | 0.6323 | 0.3966 | 0.6708 | 0.5281 | 45.9 |
| medium-p-mixed | train | 0.5185 | 0.3587 | 0.3587 | 0.1675 | 0.5695 | 0.3214 | 0.6445 | 0.4789 | 66.6 |
| medium-p-sampled | valid | 0.8220 | 0.4333 | 0.4333 | 0.2146 | 0.5646 | 0.4209 | 0.7509 | 0.5618 | 71.6 |
| medium-p-sampled | test | 0.6852 | 0.3779 | 0.3779 | 0.2267 | 0.5979 | 0.3883 | 0.6261 | 0.5849 | 66.8 |
| large-p-mixed | train | 0.8690 | 0.4962 | 0.4962 | 0.1778 | 0.6788 | 0.3076 | 0.9667 | 0.5001 | 71.0 |
| large-p-sampled | valid | 1.0474 | 0.5042 | 0.5042 | 0.2467 | 0.7873 | 0.4232 | 0.7538 | 0.5870 | 72.8 |
| large-p-sampled | test | 0.8930 | 0.5673 | 0.5673 | 0.2186 | 0.6296 | 0.3439 | 0.9427 | 0.5618 | 81.5 |

Useful comparator ladder from the same run:

- `poisson_eb` improves σ strongly over RAW but leaves a large β gap.
- `poisson_marginal_beta` provides most of the medium/large FFX gain.
- `poisson_sigma_grid` adds a smaller but consistent FFX gain and matches `current`.
- Relaxing `min_d=1` only changes small rows under the current gate structure.

Assessment
----------

- Poisson is clearly improved over RAW, but it is not yet INLA-competitive.
- FFX is the main gap. The best prototype remains about `0.10` NRMSE behind INLA on small
  rows, `0.15-0.22` behind on medium, and `0.26-0.35` behind on large.
- σ is better than RAW after EB, but still materially worse than INLA. The current grid
  does not close this because it writes back β only.
- BLUP is conservative by design. RAW/PQL BLUP fallback avoids previous regressions, but
  INLA is better in the current table, especially large mixed/test rows.
- The remaining gap likely reflects Poisson-specific β/σ/u coupling under the log link.
  More β-only optimization is unlikely to close it.

Next Directions
---------------

1. **Promote or keep testing the relaxed small-row gates.**
   `poisson_marginal_beta_min_d=1` and `poisson_sigma_grid_min_d=1` give the only large
   small-row gain so far without σ/BLUP changes. If no hidden regression appears, make this
   the default.

2. **Prototype joint small-q σ+β candidates.**
   The σ grid helps more than β-only AGQ, which suggests σ/β coupling matters. Try a tiny
   candidate set that updates σ and β together under the AGQ target. Initially write back β
   only; write back σ only if σ accuracy improves cleanly.

3. **Revisit q > 2 covariance-aware marginal β.**
   Medium/large failures are partly rows outside the current `q <= 2` σ grid. Full-`Ψ_lap`
   marginal offsets are implemented as an opt-in path; test them only after small-row
   behavior is stable.

4. **Keep fixed-budget Laplace-PIRLS as a comparator, not a default.**
   A specialized joint β/u/Σ Newton + moment update could be the right Poisson-specific
   direction, but it is a larger implementation. Build it behind a method flag only if
   smaller joint σ+β patches fail.

Low-Priority Or Rejected
------------------------

- Full PyTorch INLA, EP, broad grids, and multi-start optimization: too much complexity for
  the target analytical path.
- Bernoulli separation logic: not applicable.
- Direct broad BLUP replacement: previously regressed most cells.
- Direct σ writeback from the existing σ grid: improved FFX but regressed σ.
- β-only AGQ optimizer: small accuracy gain for much higher runtime.
- Full 8k Poisson benchmark: postpone until one more meaningful accuracy patch lands.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods raw poisson_eb poisson_marginal_beta poisson_sigma_grid current \
    --sizes small medium large --batch-size 32 --max-datasets 1000

uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current --sizes small medium large \
    --batch-size 32 --max-datasets 1000 \
    --poisson-marginal-beta-min-d 1 --poisson-sigma-grid-min-d 1

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
