Poisson GLMM Plan
=================

Last updated: 2026-05-19

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
| small-p-mixed | train | 0.4269 | 0.2261 | 0.5272 | 0.4242 | 0.5713 | 0.5210 |
| small-p-sampled | valid | 0.6375 | 0.2589 | 0.5645 | 0.4597 | 0.5823 | 0.5364 |
| small-p-sampled | test | 0.5429 | 0.2369 | 0.6323 | 0.4375 | 0.6708 | 0.5326 |
| medium-p-mixed | train | 0.3587 | 0.1901 | 0.5695 | 0.3706 | 0.6445 | 0.5113 |
| medium-p-sampled | valid | 0.4333 | 0.2394 | 0.5646 | 0.4556 | 0.7509 | 0.5568 |
| medium-p-sampled | test | 0.3779 | 0.2446 | 0.5979 | 0.3592 | 0.6261 | 0.5491 |
| large-p-mixed | train | 0.4962 | 0.1974 | 0.6788 | 0.3416 | 0.9667 | 0.5316 |
| large-p-sampled | valid | 0.5042 | 0.2716 | 0.7873 | 0.4563 | 0.7538 | 0.6062 |
| large-p-sampled | test | 0.5673 | 0.2340 | 0.6296 | 0.3713 | 0.9427 | 0.5447 |
| huge-p-mixed | train | 0.7143 | pending | 0.7202 | pending | 1.3458 | pending |
| huge-p-sampled | valid | 1.0549 | 0.2720 | 1.1463 | 0.4048 | 1.8398 | 0.6035 |
| huge-p-sampled | test | 0.7928 | 0.2579 | 0.8989 | 0.3961 | 1.2842 | 0.6051 |

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

Next Diagnostic
---------------

Run a row-level residual-gap diagnostic on the first-1000 rows with INLA references. The
diagnostic should compare current default vs INLA and bucket rows by:

- `d`, `q`, `n`, `m`, and `q > 2`;
- count shape: `y_mean`, `y_var`, `y_max`, zero fraction, high-count tail fraction;
- EB/grid diagnostics: EB accept, β accept, grid gate, grid accept, selected grid scale,
  σ cap;
- error source: β RMSE, σ RMSE, BLUP RMSE, and whether INLA mainly differs by β, σ, or both.

Primary question: are the large/huge INLA gaps concentrated in rows the current grid does
not cover or accept?

Candidate Patch
---------------

Choose the patch only after the diagnostic:

- If `q > 2` rows dominate: add a guarded `q == 3` grid with fewer σ candidates and the
  same AGQ accept target. Keep it β-only first.
- If high-count tails dominate: add a tail-gated β correction using stronger marginal
  mean shrinkage or additional downward σ-offset candidates, again β-only.
- If non-accepted grid rows dominate but INLA is better: inspect the AGQ target and add a
  stricter INLA-direction diagnostic before changing the default accept rule.
- If σ error explains the residual gap: prototype a guarded σ write-back path only for
  rows where the AGQ target and σ prior both support it. This is lower priority because
  direct σ write-back already regressed aggregate σ.

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

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
