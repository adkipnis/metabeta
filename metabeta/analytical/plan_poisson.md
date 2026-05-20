Poisson GLMM Plan
=================

Last updated: 2026-05-20 (conservative PIRLS sigma-grid guard enabled)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA is a
reference implementation only; R-INLA-as-backend and full PyTorch INLA remain out of
scope. The next retained path should be simple enough to trust and useful as context for
downstream models.

The current EB/grid path improved strongly over RAW/PQL, but the remaining INLA gap looks
like missing joint β/u/σ geometry rather than a missing scalar correction. A fixed-budget
diagonal-Σ Laplace-PIRLS prototype improves small-row σ and BLUP relative to the relaxed
EB/grid prototype. Adding marginal-mean β correction helps FFX, and a full-candidate sigma
grid around PIRLS closes a substantial part of the remaining small/medium FFX gap by
writing back β, σ, and BLUPs together.

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

Best pre-PIRLS prototype:

```python
glmm(
    ...,
    poisson_marginal_beta_min_d=1,
    poisson_sigma_grid_min_d=1,
)
```

This relaxes the two `d >= 5` gates. It substantially improves small rows and leaves
medium/large unchanged because those gates already fired.

Opt-in joint prototype:

```python
glmm(..., poisson_laplace_pirls_diag=True)
```

This runs fixed-budget joint β/u PIRLS steps with diagonal σ, updates σ from posterior
second moments, and accepts/rejects the full β/σ/BLUP candidate by the diagonal Laplace
target.

Best current prototype:

```python
glmm(
    ...,
    poisson_laplace_pirls_diag=True,
    poisson_marginal_beta=True,
    poisson_marginal_beta_min_d=1,
    poisson_laplace_pirls_sigma_grid=True,
    poisson_sigma_grid=False,
)
```

In the benchmark this is exposed as `poisson_laplace_pirls_full_grid`. It uses PIRLS for
σ/BLUP geometry, applies the marginal-mean β correction, then evaluates a conservative
diagonal σ grid with scales `(0.5, 0.75, 1.0)`. Each σ candidate gets a few fixed-σ joint
β/u PIRLS steps and is accepted by the same diagonal Laplace target. Unlike the older σ
grid, it writes back the full candidate. The grid deliberately allows shrinkage or fixed-σ
β/u re-synchronization only; inflation scales are disabled by default after producing rare
large σ outliers. It should remain opt-in until large rows are checked.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU runs on 2026-05-20. Lower NRMSE is better.
INLA values are the current first-1000 diagonal R-INLA references. "Relaxed current" means
`current` with `poisson_marginal_beta_min_d=1` and `poisson_sigma_grid_min_d=1`.

| Dataset | part | RAW FFX | default FFX | relaxed FFX | PIRLS+β FFX | full-grid FFX | INLA FFX | relaxed σ | PIRLS+β σ | full-grid σ | INLA σ | relaxed BLUP | PIRLS+β BLUP | full-grid BLUP | INLA BLUP | full-grid ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.4806 | 0.4269 | 0.2839 | 0.2585 | 0.2214 | 0.1835 | 0.5272 | 0.4465 | 0.4529 | 0.3404 | 0.5713 | 0.5413 | 0.5245 | 0.4936 | 37.9 |
| small-p-sampled | valid | 0.7351 | 0.6375 | 0.3277 | 0.3182 | 0.2523 | 0.2276 | 0.5645 | 0.4955 | 0.4919 | 0.4356 | 0.5823 | 0.5539 | 0.5384 | 0.5309 | 34.2 |
| small-p-sampled | test | 0.6525 | 0.5429 | 0.2982 | 0.2797 | 0.2205 | 0.1997 | 0.6323 | 0.4841 | 0.4618 | 0.3966 | 0.6708 | 0.5533 | 0.5270 | 0.5281 | 36.4 |

Medium/large first-1000 validation:

| Dataset | part | current FFX | PIRLS+β FFX | full-grid FFX | INLA FFX | current σ | PIRLS+β σ | full-grid σ | INLA σ | current BLUP | PIRLS+β BLUP | full-grid BLUP | INLA BLUP | current ms/ds | full-grid ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium-p-mixed | train | 0.3587 | 0.3391 | 0.2670 | 0.1675 | 0.5695 | 0.4798 | 0.4360 | 0.3214 | 0.6445 | 0.5392 | 0.5199 | 0.4789 | 69.1 | 66.7 |
| medium-p-sampled | valid | 0.4333 | 0.3989 | 0.3385 | 0.2146 | 0.5646 | 0.5274 | 0.5229 | 0.4209 | 0.7509 | 0.6896 | 0.6334 | 0.5618 | 72.3 | 68.8 |
| medium-p-sampled | test | 0.3779 | 0.3565 | 0.2968 | 0.2267 | 0.5979 | 0.5056 | 0.5016 | 0.3883 | 0.6261 | 0.5848 | 0.5603 | 0.5849 | 67.6 | 64.6 |
| large-p-mixed | train | 0.4962 | 0.4723 | n/t | 0.1778 | 0.6788 | 0.7117 | n/t | 0.3076 | 0.9667 | 0.7726 | n/t | 0.5001 | 79.9 | n/t |
| large-p-sampled | valid | 0.5042 | 0.5161 | n/t | 0.2467 | 0.7873 | 0.5977 | n/t | 0.4232 | 0.7538 | 0.7006 | n/t | 0.5870 | 78.0 | n/t |
| large-p-sampled | test | 0.5673 | 0.5386 | n/t | 0.2186 | 0.6296 | 0.7846 | n/t | 0.3439 | 0.9427 | 0.8512 | n/t | 0.5618 | 83.1 | n/t |

Useful comparator ladder from the same run:

- `poisson_eb` improves σ strongly over RAW but leaves a large β gap.
- `poisson_marginal_beta` provides most of the medium/large FFX gain.
- `poisson_sigma_grid` adds a smaller but consistent FFX gain and matches `current`.
- Relaxing `min_d=1` only changes small rows under the current gate structure.
- `poisson_laplace_pirls_diag` improves small-row σ and BLUP versus relaxed current, and
  is faster than the relaxed grid path in these runs.
- `poisson_laplace_pirls_beta` keeps the PIRLS σ/BLUP gains and beats relaxed current FFX
  on all small rows without invoking the σ grid.
- On medium rows, `poisson_laplace_pirls_beta` improves FFX, σ, BLUP, and runtime versus
  current in all three cells.
- `poisson_laplace_pirls_full_grid` is the first Poisson patch that substantially closes
  the remaining small/medium FFX gap. It improves FFX on every small and medium row and
  improves BLUP on every small and medium row. σ improves on small-test and medium-mixed,
  is roughly neutral on medium-test, and regresses on small-mixed/small-valid/medium-valid.
- On large rows, `poisson_laplace_pirls_beta` improves FFX in mixed/test and BLUP in all
  cells, but regresses FFX slightly on sampled-valid and regresses σ on mixed/test. The
  full-grid path has not been checked on large rows yet.

Full-candidate sigma-grid diagnostic:

```bash
uv run python -u experiments/analytical/glmm_poisson_pirls_grid_diagnostic.py \
    --sizes small medium --max-datasets 1000 --batch-size 32
```

The default diagnostic now uses conservative scales `(0.5, 0.75, 1.0)`. Across the 6000
small/medium diagnostic rows, full-grid acceptance is `0.982-0.999` by cell. About 82% of
accepted rows select scale `1.0`, meaning the grid mostly accepts a fixed-σ β/u re-sync
after the marginal β correction. True σ scaling is less common and only shrinks σ:

| scale | rows | base FFX | grid FFX | base σ | grid σ | base BLUP | grid BLUP | comment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 179 | 0.4058 | 0.2918 | 0.8973 | 0.7484 | 0.7020 | 0.6804 | large gains, σ improves |
| 0.75 | 856 | 0.3024 | 0.2218 | 0.6413 | 0.5946 | 0.6741 | 0.6398 | useful shrinkage |
| 1.0 | 4922 | 0.3321 | 0.2736 | 0.4642 | 0.4642 | 0.5714 | 0.5431 | β/u re-sync only |
| rejected | 43 | 1.4224 | 1.4224 | 0.4753 | 0.4753 | 0.5760 | 0.5760 | unchanged |

Row-level tradeoff rates:

| cell | FFX win | σ loss | FFX win + σ loss | mean ΔFFX | mean Δσ | mean ΔBLUP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed train | 0.565 | 0.043 | 0.025 | -0.0110 | -0.0026 | -0.0112 |
| small-p-sampled valid | 0.576 | 0.047 | 0.027 | -0.0193 | -0.0024 | -0.0103 |
| small-p-sampled test | 0.599 | 0.042 | 0.028 | -0.0171 | -0.0030 | -0.0124 |
| medium-p-mixed train | 0.684 | 0.057 | 0.041 | -0.0250 | -0.0066 | -0.0125 |
| medium-p-sampled valid | 0.689 | 0.051 | 0.042 | -0.0284 | -0.0032 | -0.0181 |
| medium-p-sampled test | 0.692 | 0.057 | 0.046 | -0.0249 | -0.0055 | -0.0141 |
| all | 0.634 | 0.050 | 0.035 | -0.0209 | -0.0039 | -0.0131 |

Interpretation: the grid is primarily a β/u re-synchronization device, not a broad σ
inflation strategy. Exploratory inflation scales showed that `2.0` was almost never chosen
but dominated the worst σ tradeoffs, while `1.333` preserved FFX but doubled the σ-loss
row rate. Dropping both inflation scales leaves FFX essentially unchanged, improves mean σ,
and cuts σ-loss rows from about `12.3%` to `5.0%`.

Assessment
----------

- Poisson is clearly improved over RAW, but it is not yet INLA-competitive.
- FFX is the main gap. The full-candidate grid reduces the residual small-row FFX gap to
  about `0.020-0.038` NRMSE and the medium-row gap to about `0.070-0.116`. Large rows are
  still represented by the older PIRLS+β numbers and remain about `0.29-0.32` behind INLA.
- The first PIRLS prototype confirms the joint-geometry hypothesis for σ/BLUP. The hybrid
  result confirms that PIRLS geometry feeds the marginal-mean β correction better than the
  old EB/grid geometry.
- σ is better than RAW after EB, but still materially worse than INLA. The conservative
  full-candidate grid avoids broad σ inflation and improved σ versus the first full-grid
  diagnostic, but it still trails INLA.
- BLUP is conservative by design. RAW/PQL BLUP fallback avoids previous regressions, but
  INLA is better in the current table, especially large mixed/test rows.
- The remaining gap likely reflects Poisson-specific β/σ/u coupling under the log link.
  More β-only optimization is unlikely to close it.
- Wrong σ directly distorts the marginal mean through the log link. If σ is too small or
  too large, β-only marginal corrections tend to under- or over-correct.
- The old σ grid is especially limited because it uses σ candidates to improve β, then
  writes back β only. The new full-candidate grid directly addresses this by writing back
  β, σ, and BLUPs from the winning Laplace-scored candidate.

Next Directions
---------------

1. **Validate the conservative full-candidate grid on large rows without tuning.**
   Do not tune against large yet. First run the same opt-in method with conservative
   scales `(0.5, 0.75, 1.0)` to see whether the small/medium FFX gain survives and whether
   the known large-row σ regressions worsen.

2. **Tune the PIRLS covariance update only after full-grid validation.**
   The current diagonal update is:

   ```text
   sigma_j^2 <- (sum_g (u_gj^2 + diag(A_g^-1)_j) + nu0 * sigma0_j^2) / (m + nu0)
   ```

   Useful knobs are stronger `nu0`, lower log-σ blend, smaller PIRLS damping, and final
   fixed-σ PIRLS steps. Tune only against the large σ regressions, not as a broad grid.

3. **Keep accept/reject by one coherent cheap Laplace target.**
   Compare joint candidates with the same approximate Laplace objective:

   ```text
   J = poisson_nll(y | beta, u)
       + 0.5 * sum_g u_g' Sigma^-1 u_g
       + 0.5 * m * log|Sigma|
       + 0.5 * sum_g log|Z_g' W_g Z_g + Sigma^-1|
       + priors
   ```

   Keep sanity guards for finite values, bounded η/μ, σ floors/caps, and catastrophic β
   jumps. Do not use RAW/PQL BLUP fallback inside the joint candidate; fallback only if the
   whole candidate fails or loses.

4. **Only after guarded diagonal PIRLS+full-grid is stable, test full Σ.**
   Full covariance is cheap for `q <= 5`, but it adds instability risk. Test it inside the
   joint Laplace-PIRLS solver, not as another posthoc marginal β correction.

5. **Retire the old β-only sigma grid if full-candidate grid holds up.**
   The older grid is slower than PIRLS+β and less coherent than full-candidate writeback.

Low-Priority Or Rejected
------------------------

- Full PyTorch INLA, EP, broad grids, and multi-start optimization: too much complexity for
  the target analytical path.
- Bernoulli separation logic: not applicable.
- Direct broad BLUP replacement: previously regressed most cells.
- Direct σ writeback from the existing σ grid: improved FFX but regressed σ when used as a
  posthoc patch. Revisit only as full-candidate rescue after joint PIRLS.
- β-only AGQ optimizer: small accuracy gain for much higher runtime.
- More relaxed gates and more β-only marginal correction tuning: useful as ablations, but
  unlikely to close the INLA gap.
- Full-`Ψ_lap` marginal β correction as posthoc default: lower priority than testing full
  covariance inside a joint solver.
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

# Medium/large validation for the current PIRLS+β prototype:
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_laplace_pirls_diag poisson_laplace_pirls_beta \
    --sizes medium large --batch-size 32 --max-datasets 1000 \
    --poisson-marginal-beta-min-d 1 --poisson-sigma-grid-min-d 1

# Conservative full-candidate PIRLS grid:
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods poisson_laplace_pirls_full_grid \
    --sizes small medium large --batch-size 32 --max-datasets 1000 \
    --poisson-marginal-beta-min-d 1

# Small-row relaxed-current versus diagonal PIRLS+β check:
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_laplace_pirls_diag poisson_laplace_pirls_beta \
    --combos small-p-mixed:train:1 small-p-sampled:valid small-p-sampled:test \
    --batch-size 32 --max-datasets 1000 \
    --poisson-marginal-beta-min-d 1 --poisson-sigma-grid-min-d 1

# Full-candidate PIRLS sigma-grid scale and tradeoff diagnostic:
uv run python -u experiments/analytical/glmm_poisson_pirls_grid_diagnostic.py \
    --sizes small medium --max-datasets 1000 --batch-size 32

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
