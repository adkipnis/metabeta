Poisson GLMM Plan
=================

Last updated: 2026-05-20 (diagnostic complete)

Goal
----

Build a fast, high-accuracy analytical Poisson GLMM estimator for `glmm()`. The retained
path should be prior-aware, batched, and simple enough to trust. R-INLA remains a reference
only; full INLA or R-INLA-as-backend is out of scope.

Current Default
---------------

The default Poisson path is now:

1. RAW/PQL initialization.
2. Diagonal Poisson Laplace EB over β and diagonal σ_rfx.
3. Per-row accept gate against RAW/PQL.
4. RAW/PQL BLUP fallback, because direct Poisson EB BLUP modes regressed most benchmark
   cells.
5. Accepted-row σ cap at `2.5 * tau_rfx` for effective `d >= 5`.
6. Gated marginal-mean β correction for `d >= 5`.
7. Targeted σ-offset grid for `d >= 5` and `q <= 2`.

The σ-offset grid is deliberately β-only: it tries a few σ scales to generate alternative
marginal-mean β candidates, accepts by a cheap adaptive Gauss-Hermite marginal posterior,
then writes back β while keeping EB σ and RAW/PQL BLUP unchanged.

Useful comparators:

```python
# RAW/PQL
glmm(..., map_refine=False)

# EB only
glmm(..., poisson_laplace_eb='poisson_eb', poisson_marginal_beta=False)

# EB + marginal β, no σ-offset grid
glmm(..., poisson_laplace_eb='poisson_eb', poisson_marginal_beta=True,
     poisson_sigma_grid=False)
```

Current Evidence
----------------

First 1000 rows per cell. Lower NRMSE is better. Analytical timing from the latest run is
not treated as decisive while background INLA jobs are consuming CPU.

| Dataset | part | default FFX | INLA FFX | default σ | INLA σ | default BLUP | INLA BLUP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.4269 | 0.1835 | 0.5272 | 0.3404 | 0.5713 | 0.4936 |
| small-p-sampled | valid | 0.6375 | 0.2276 | 0.5645 | 0.4356 | 0.5823 | 0.5309 |
| small-p-sampled | test | 0.5429 | 0.1997 | 0.6323 | 0.3966 | 0.6708 | 0.5281 |
| medium-p-mixed | train | 0.3587 | 0.1675 | 0.5695 | 0.3214 | 0.6445 | 0.4789 |
| medium-p-sampled | valid | 0.4333 | 0.2146 | 0.5646 | 0.4209 | 0.7509 | 0.5618 |
| medium-p-sampled | test | 0.3779 | 0.2267 | 0.5979 | 0.3883 | 0.6261 | 0.5849 |

What This Means
---------------

- The Bernoulli-style EB transplant helped Poisson, but not enough. Poisson still needs
  log-link-specific marginal mean calibration.
- Marginal β and the σ-offset grid are the retained Poisson-specific fixes. They improve
  medium/large/huge FFX while leaving σ and BLUP unchanged.
- The largest remaining gaps are now not a reason to add broad machinery. They should be
  localized first: likely candidates are `q > 2`, high-count tails, non-accepted grid rows,
  or cases where the β-only grid cannot compensate for a materially wrong σ.
- BLUP remains intentionally conservative. Changing Poisson BLUPs should be postponed
  until FFX/σ diagnostics show a clear, contained failure mode.

Rejected Or Low-Priority Work
-----------------------------

- Full PyTorch INLA, EP, broad sigma grids, and multi-start optimization: too many moving
  parts for the target path.
- Bernoulli separation logic: not applicable to Poisson.
- Directly writing σ-grid candidates back to output: tested and rejected because FFX
  improved but σ regressed.
- Broad BLUP mode replacement: tested and rejected because it regressed most cells.
- Full 8k Poisson benchmark: postpone until the residual INLA gap has been diagnosed and
  one more targeted patch has either passed or been rejected.

Diagnostic Findings (2026-05-20)
--------------------------------

Row-level direction diagnostic run on first-1000 INLA-matched rows per cell using
`glmm_poisson_inla_direction_diagnostic.py` with `sortish=False` (bug fix: the old
diagnostic used `sortish=True` which scrambled the INLA index mapping and inflated INLA
RMSE; the plan table from `glmm_inla_comparison.py` was unaffected as it already used
`sortish=False`).

Summary of raw RMSE and direction metrics per cell:

| Dataset | part | RAW β | EB β | INLA β | β gain | RAW σ | EB σ | INLA σ | σ gain | BLUP gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.095 | 0.104 | 0.078 | −0.009 | 0.135 | 0.084 | 0.067 | +0.052 | +0.021 |
| small-p-sampled | valid | 0.143 | 0.133 | 0.101 | +0.010 | 0.153 | 0.099 | 0.083 | +0.055 | +0.022 |
| small-p-sampled | test | 0.123 | 0.118 | 0.091 | +0.005 | 0.148 | 0.094 | 0.080 | +0.054 | +0.023 |
| medium-p-mixed | train | 0.105 | 0.113 | 0.072 | −0.007 | 0.157 | 0.088 | 0.058 | +0.069 | +0.032 |
| medium-p-sampled | valid | 0.167 | 0.137 | 0.095 | +0.030 | 0.180 | 0.099 | 0.073 | +0.081 | +0.034 |
| medium-p-sampled | test | 0.162 | 0.141 | 0.096 | +0.022 | 0.190 | 0.103 | 0.071 | +0.088 | +0.037 |

Key findings:

**1. q > 2 rows are the primary failure mode on medium datasets.**
The sigma grid only fires for `d >= 5` and `q <= 2` (sg_gate). On medium (d=5–8), the
q > 2 bucket (20–28% of rows) has FFX gap 2–5× worse than q ≤ 2, EB moves β strongly
*away* from INLA (toward% < 50%), and sigma cap rate is 14–19% vs 5–6% for q ≤ 2.
The marginal beta gate fires for all medium rows (mb_gate=1), but without the sigma grid,
the β-only correction has no alternative σ scales to work with for q > 2.

| Group (medium) | N% | FFX gain | FFX gap | toward% | σ cap% |
| --- | ---: | ---: | ---: | ---: | ---: |
| sg_gate=1 (q≤2) | ~75% | +0.003 to +0.033 | 0.022–0.035 | 44–50% | 5–6% |
| sg_gate=0 (q>2) | ~25% | −0.017 to −0.038 | 0.075–0.118 | — | 14–19% |

**2. Small datasets (d≤4): no grids fire, σ improves, β is near-neutral.**
All small-p rows have d≤4, so both mb_gate and sg_gate are 0. EB consistently improves σ
(+0.052–0.055 true gain). FFX effect is near zero (−0.009 to +0.010 true gain). The
remaining small-p INLA gap is not grid-related; it is a general EB β bias, mostly on
high-count rows (mean_y ≥ 2, gain −0.011 to −0.025).

**3. High-count rows (mean_y ≥ 2) are a secondary failure mode across all datasets.**
On all six cells, mean_y ≥ 2 shows the worst FFX gains (most negative) and lowest accept
rates (~80–87%). For medium, this bucket overlaps heavily with q > 2 rows.

**4. BLUP is unchanged and slightly better than INLA.** The fallback is active on all
rows (gain = 0, gain% = 0). The BLUP gap is positive (EB slightly closer to truth than
INLA) across all cells, confirming the fallback is the right call.

**5. σ improvement is consistent and moves toward INLA.** On all six cells, sigma toward%
is 69–78% and median dist gain is 0.59–0.69. The EB σ improvement is real and should not
be disrupted by any new patch.

Answer to the primary diagnostic question: the INLA gap on medium-p datasets is
concentrated in `q > 2` rows that the current sigma grid does not cover (sg_gate=0). This
unambiguously selects the first candidate patch.

Next Patch
----------

Add a guarded `q > 2` marginal-beta correction, β-only, modelled on the existing
`q <= 2` sigma-offset grid:

- Gate: `d >= 5` and `q > 2` (i.e., where mb_gate=1 and sg_gate=0).
- Try a small set of σ scales (e.g., 3–4 candidates: 0.5×, 0.75×, 1.0×, 1.25× EB σ).
- Accept by AGQ marginal posterior, write back β only; keep EB σ and RAW/PQL BLUP.
- Do not touch q ≤ 2 rows; their path is already better and σ-grid-covered.

Prototype the patch, run the required benchmark on small+medium cells, and run the
direction diagnostic on medium-p-mixed/train to confirm β moves toward INLA without
regressing σ.

Commands
--------

```bash
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods raw current poisson_eb poisson_marginal_beta poisson_sigma_grid \
    --combos small-p-mixed:train:2 small-p-sampled:valid small-p-sampled:test \
        medium-p-mixed:train:2 medium-p-sampled:valid medium-p-sampled:test \
        large-p-mixed:train:2 large-p-sampled:valid large-p-sampled:test \
        huge-p-mixed:train:2 huge-p-sampled:valid huge-p-sampled:test \
    --batch-size 32 --max-datasets 1000

uv run python -u experiments/analytical/glmm_inla_comparison.py \
    --data-ids medium-p-sampled --partition valid \
    --n-inla 1000 --n-total 1000 --analytical-methods raw,current

# Poisson direction diagnostic (precomputed INLA fits; --n-epochs 1 for all partitions)
uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids small-p-mixed --partition train --n-inla 1000 --n-epochs 1

uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids small-p-sampled --partition valid --n-inla 1000 --n-epochs 1

uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids small-p-sampled --partition test --n-inla 1000 --n-epochs 1

uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids medium-p-mixed --partition train --n-inla 1000 --n-epochs 1

uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids medium-p-sampled --partition valid --n-inla 1000 --n-epochs 1

uv run python -u experiments/analytical/glmm_poisson_inla_direction_diagnostic.py \
    --data-ids medium-p-sampled --partition test --n-inla 1000 --n-epochs 1

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
