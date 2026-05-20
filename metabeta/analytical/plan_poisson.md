Poisson GLMM Plan
=================

Last updated: 2026-05-20 (PIRLS + marginal-mean β hybrid tested through large rows)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA is a
reference implementation only; R-INLA-as-backend and full PyTorch INLA remain out of
scope. The next retained path should be simple enough to trust and useful as context for
downstream models.

The current EB/grid path improved strongly over RAW/PQL, but the remaining INLA gap looks
like missing joint β/u/σ geometry rather than a missing scalar correction. A fixed-budget
diagonal-Σ Laplace-PIRLS prototype improves small-row σ and BLUP relative to the relaxed
EB/grid prototype. Adding the marginal-mean β correction on top also improves small-row
FFX beyond relaxed current while avoiding the slower σ grid.

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
    poisson_sigma_grid=False,
)
```

In the benchmark this is exposed as `poisson_laplace_pirls_beta`. It uses PIRLS for
σ/BLUP geometry and then applies the marginal-mean β correction. It should remain opt-in
until medium/large rows are checked.

Current Evidence
----------------

First 1000 rows per cell, sequential CPU runs on 2026-05-20. Lower NRMSE is better.
INLA values are the current first-1000 diagonal R-INLA references. "Relaxed current" means
`current` with `poisson_marginal_beta_min_d=1` and `poisson_sigma_grid_min_d=1`.

| Dataset | part | RAW FFX | default FFX | relaxed FFX | PIRLS FFX | PIRLS+β FFX | INLA FFX | relaxed σ | PIRLS+β σ | INLA σ | relaxed BLUP | PIRLS+β BLUP | INLA BLUP | PIRLS+β ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small-p-mixed | train | 0.4806 | 0.4269 | 0.2839 | 0.3054 | 0.2585 | 0.1835 | 0.5272 | 0.4465 | 0.3404 | 0.5713 | 0.5413 | 0.4936 | 31.4 |
| small-p-sampled | valid | 0.7351 | 0.6375 | 0.3277 | 0.4354 | 0.3182 | 0.2276 | 0.5645 | 0.4955 | 0.4356 | 0.5823 | 0.5539 | 0.5309 | 28.9 |
| small-p-sampled | test | 0.6525 | 0.5429 | 0.2982 | 0.3684 | 0.2797 | 0.1997 | 0.6323 | 0.4841 | 0.3966 | 0.6708 | 0.5533 | 0.5281 | 30.4 |

Medium/large first-1000 validation:

| Dataset | part | current FFX | PIRLS FFX | PIRLS+β FFX | INLA FFX | current σ | PIRLS+β σ | INLA σ | current BLUP | PIRLS+β BLUP | INLA BLUP | current ms/ds | PIRLS+β ms/ds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium-p-mixed | train | 0.3587 | 0.3529 | 0.3391 | 0.1675 | 0.5695 | 0.4798 | 0.3214 | 0.6445 | 0.5392 | 0.4789 | 73.8 | 59.2 |
| medium-p-sampled | valid | 0.4333 | 0.5457 | 0.3989 | 0.2146 | 0.5646 | 0.5274 | 0.4209 | 0.7509 | 0.6896 | 0.5618 | 73.1 | 57.7 |
| medium-p-sampled | test | 0.3779 | 0.4247 | 0.3565 | 0.2267 | 0.5979 | 0.5056 | 0.3883 | 0.6261 | 0.5848 | 0.5849 | 72.5 | 56.8 |
| large-p-mixed | train | 0.4962 | 0.5757 | 0.4723 | 0.1778 | 0.6788 | 0.7117 | 0.3076 | 0.9667 | 0.7726 | 0.5001 | 79.9 | 66.9 |
| large-p-sampled | valid | 0.5042 | 0.7046 | 0.5161 | 0.2467 | 0.7873 | 0.5977 | 0.4232 | 0.7538 | 0.7006 | 0.5870 | 78.0 | 65.3 |
| large-p-sampled | test | 0.5673 | 0.6226 | 0.5386 | 0.2186 | 0.6296 | 0.7846 | 0.3439 | 0.9427 | 0.8512 | 0.5618 | 83.1 | 72.2 |

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
- On large rows, `poisson_laplace_pirls_beta` improves FFX in mixed/test and BLUP in all
  cells, but regresses FFX slightly on sampled-valid and regresses σ on mixed/test.

Assessment
----------

- Poisson is clearly improved over RAW, but it is not yet INLA-competitive.
- FFX is the main gap. The PIRLS+β hybrid remains about `0.075-0.105` NRMSE behind INLA
  on small rows, `0.13-0.18` behind on medium, and `0.29-0.32` behind on large.
- The first PIRLS prototype confirms the joint-geometry hypothesis for σ/BLUP. The hybrid
  result confirms that PIRLS geometry feeds the marginal-mean β correction better than the
  old EB/grid geometry.
- σ is better than RAW after EB, but still materially worse than INLA. PIRLS+β helps σ on
  small and medium rows, but large mixed/test show σ regressions, so a broad default needs
  either a σ guard or a tuned covariance update.
- BLUP is conservative by design. RAW/PQL BLUP fallback avoids previous regressions, but
  INLA is better in the current table, especially large mixed/test rows.
- The remaining gap likely reflects Poisson-specific β/σ/u coupling under the log link.
  More β-only optimization is unlikely to close it.
- Wrong σ directly distorts the marginal mean through the log link. If σ is too small or
  too large, β-only marginal corrections tend to under- or over-correct.
- The current σ grid is especially limited because it uses σ candidates to improve β, then
  writes back β only. This does not let β, BLUPs, and σ repeatedly adapt to each other.

Next Directions
---------------

1. **Add a lightweight PIRLS+β guard before defaulting.**
   PIRLS+β is clearly better on small/medium, but large rows are not uniformly safe. Use
   cheap internal diagnostics to reject the PIRLS σ/BLUP writeback when the diagonal
   covariance update looks unreliable, while still allowing the marginal β correction when
   it improves the existing target. Candidate guards: large σ ratio to EB/current,
   Laplace-target loss, or excessive marginal correction magnitude.

2. **Tune the PIRLS covariance update against the large-row pattern.**
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

4. **Only after guarded diagonal PIRLS+β is stable, test full Σ.**
   Full covariance is cheap for `q <= 5`, but it adds instability risk. Test it inside the
   joint Laplace-PIRLS solver, not as another posthoc marginal β correction.

5. **Use sigma-grid rescue only after the joint prototype.**
   If the diagonal solver helps but has identifiable failure rows, test a targeted rescue
   that writes back the full candidate: β, σ, and BLUPs. Do not prioritize the old β-only
   grid as the main route.

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

# Small-row relaxed-current versus diagonal PIRLS+β check:
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods current poisson_laplace_pirls_diag poisson_laplace_pirls_beta \
    --combos small-p-mixed:train:1 small-p-sampled:valid small-p-sampled:test \
    --batch-size 32 --max-datasets 1000 \
    --poisson-marginal-beta-min-d 1 --poisson-sigma-grid-min-d 1

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
