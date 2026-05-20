Poisson GLMM Plan
=================

Last updated: 2026-05-20 (pivot to fixed-budget Laplace-PIRLS)

Goal
----

Build a fast, prior-aware analytical Poisson GLMM estimator for `glmm()`. R-INLA is a
reference implementation only; R-INLA-as-backend and full PyTorch INLA remain out of
scope. The next retained path should be simple enough to trust and useful as context for
downstream models.

The current EB/grid path improved strongly over RAW/PQL, but the remaining INLA gap looks
like missing joint β/u/σ geometry rather than a missing scalar correction. The next main
Poisson experiment is therefore a fixed-budget diagonal-Σ Laplace-PIRLS solver. The
existing EB/grid estimator should remain as initializer, fallback, and diagnostic baseline.

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

Status: keep this path, but do not keep stacking posthoc β-only corrections as the primary
route. Its main value now is as a strong cheap baseline and a stable initializer/fallback
for a joint Poisson solver.

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
- Wrong σ directly distorts the marginal mean through the log link. If σ is too small or
  too large, β-only marginal corrections tend to under- or over-correct.
- The current σ grid is especially limited because it uses σ candidates to improve β, then
  writes back β only. This does not let β, BLUPs, and σ repeatedly adapt to each other.

Next Directions
---------------

1. **Implement opt-in diagonal fixed-budget Laplace-PIRLS.**
   This is the decisive next prototype. For fixed diagonal Σ, jointly update β and group
   modes `u_g` with Schur-complement PIRLS/Newton steps, then update σ from posterior
   second moments:

   ```text
   sigma_j^2 <- (sum_g (u_gj^2 + diag(A_g^-1)_j) + nu0 * sigma0_j^2) / (m + nu0)
   ```

   Use a small fixed budget first: 3-4 outer σ updates, 1-2 PIRLS steps per outer, and 1-2
   final β/u steps with σ fixed. Start diagonal only.

2. **Accept/reject by one coherent cheap Laplace target.**
   Compare the joint candidate against the current EB/grid path with the same approximate
   Laplace objective:

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

3. **Benchmark the new method as a separate method flag.**
   Compare `raw`, `current`, `poisson_laplace_pirls_diag`, and INLA on first 1k
   small/medium/large rows. Track FFX, σ, BLUP, runtime, finite/fallback rate, and Laplace
   target deltas. The key success criterion is reducing the medium/large FFX gap without
   major σ or BLUP regression.

4. **Only after diagonal works, test full Σ.**
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

# Intended next benchmark once the Laplace-PIRLS method is wired in:
uv run python -u experiments/analytical/glmm_required_benchmark.py \
    --family p --methods raw current poisson_laplace_pirls_diag \
    --sizes small medium large --batch-size 32 --max-datasets 1000

uv run pytest tests/utils/test_glmm.py
uv run blue --check --diff metabeta/analytical/fit.py metabeta/analytical/glmm \
    experiments/analytical
```
