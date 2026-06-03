# Parameter Prior Coverage Plan

## Current Finding

The real-data NUTS reference and `testset_overview.py` diagnostics suggest that the
normal-family synthetic data is too signal dominated after outcome normalization.
Generated training data puts too much mass near `r_squared = 1.0`, while the real
continuous reference fits place substantially more mass on residual variation.

This is not controlled by Bambi defaults. The generated parameter coverage comes
from `metabeta/simulation/prior.py::hypersample()` and the subsequent normalization
inside `metabeta/simulation/simulator.py::Simulator.sample()`.

## Why Direct Hyperprior Edits Are Not Enough

For normal data, the simulator samples raw fixed effects, random effects, and
residual noise, then divides `y` and all scale parameters by `sd(y)`. Because of
that normalization, increasing all raw scales together mostly cancels out. The
post-normalization quantities used for training depend on relative variance:

```text
sd(y)^2 ~= Var(X beta + Z b) + sigma_eps_raw^2
sigma_eps_stored ~= sigma_eps_raw / sd(y)
r_squared = 1 - sigma_eps_stored^2
```

So the key target is the residual share of total outcome variance, not just the
absolute raw value of `tau_eps`.

## Diagnostics Before Any Hyperprior Change

1. Regenerate a small normal training sample after the shape-sampling fix.
2. Run `experiments/simulation/testset_overview.py`.
3. Compare generated coverage against strict and liberal real-NUTS references:
   - `sigma_eps`
   - `sigma_rfx`
   - `r_squared`
   - fixed-effect magnitude
   - `eta_rfx` / correlation coverage for `q > 1`
4. Confirm whether the high `r_squared` mismatch persists after shape changes.

## Conservative First Pass

If the mismatch persists, adjust only normal-family hyperparameter sampling:

1. Reduce fixed-effect signal modestly:
   - lower the normal-family `tau_ffx` mode or concentration in `hypersample()`.
2. Raise residual share:
   - increase the typical normal-family `tau_eps` relative to `tau_ffx` and
     `tau_rfx`.
3. Keep random-effect SD coverage broad:
   - avoid shrinking `tau_rfx` too strongly, because the real-NUTS reference
     still shows meaningful between-group variation.
4. Re-run the overview and compare post-normalization quantities, not raw
   hyperparameters.

Acceptance target: reduce the generated mass at `r_squared > 0.95` while keeping
realistic coverage for `sigma_rfx` and fixed effects.

## Preferred Robust Approach

If conservative hyperprior tuning is unstable, introduce an explicit residual-share
sampler for normal data:

1. Sample fixed and random effects as usual.
2. Compute the latent signal `eta = X beta + Z b` for the sampled design.
3. Sample a target normalized residual SD, for example from a broad distribution
   over roughly `0.35-0.95`.
4. Set raw `sigma_eps` from the signal variance so the final normalized
   `sigma_eps` lands near that target.
5. Normalize `y` exactly as today, preserving the existing training data contract.

This controls the parameter actually seen by the NPE after normalization and should
be easier to validate than indirect `tau_eps` tuning.

Implemented variant: use a dataset-specific `R^2` cap rather than a hard target.
The cap increases with the number of fixed-effect covariates and random slopes, and
the simulator only raises `sigma_eps` when the sampled signal share exceeds that
cap. This preserves naturally low-signal draws and still allows higher `R^2` as
the available covariate space grows. When `sigma_eps` is raised, `tau_eps` is
scaled by the same factor so the stored prior context remains coherent.

## Bernoulli Case

### Finding

`real-b-reference` NUTS fits (14 real binary datasets) and testset_overview diagnostics
reveal a cumulative linear-predictor variance problem for large d:

- `hypersample()` draws `tau_ffx ~ skewedBeta(0.01, 3.0, mode=0.8)` per predictor.
- For d=13–16 (huge), `SD(eta_FFX) = sqrt(sum tau_j^2)` has mean ≈ 4.5 and reaches 6.6.
- P(|eta| > 10) at SD=6.6 is ≈ 13% per observation — irrecoverable by 20 rerolls.
- Generation logs show 26 "remained high after rerolls" warnings across all sizes,
  concentrated in large/huge batches.

From the real-b-reference NUTS posteriors:
- `sd(eta_hat)` = 0.91–3.27, mean ≈ 2.2, p95 ≈ 3.2.
- Even for d=16 (guimmun) and d=22 (guprenat), sd(eta_hat) ≤ 3.3 because real
  high-d datasets tend to have many small, sparse effects.

Unlike Normal, there is no `sigma_eps` to act as a lever — the entire signal amplitude
is the LP variance. Raising or lowering `tau_ffx` shifts the mode but does not prevent
tail draws from producing extreme cumulative LP variance at high d.

### Why Direct Hyperprior Narrowing Is Not Enough

Narrowing `tau_ffx` max from 3.0 to e.g. 1.5 would make large-d priors
conservative, but leaves small-d cases under-covered. The problem is
that cumulative variance grows with d regardless of any fixed per-predictor cap.

### Preferred Approach: LP Scale Calibration

Analogous to the Normal's `_calibrateBernoulliResidualShare`, introduce
`_calibrateBernoulliEtaScale` in `Simulator.sample()`:

1. After parameter sampling (and any rerolls), compute `eta = X beta + Z b`.
2. Sample a target LP SD cap uniformly from `[BERNOULLI_LP_SD_CAP_LOW,
   BERNOULLI_LP_SD_CAP_HIGH]` = `[2.0, 4.0]`.
3. If `sd(eta) > cap`, scale all parameters (`ffx`, `rfx`, `sigma_rfx`) and
   hyperparameters (`tau_ffx`, `nu_ffx`, `tau_rfx`) by `cap / sd(eta)`.
4. This keeps the prior context coherent: stored hyperparams reflect the
   effective prior after calibration.

Cap bounds rationale:
- Lower = 2.0: preserves high-signal datasets up to the real-data maximum.
- Upper = 4.0: at SD=4 → P(|eta| > 10) ≈ 1.2%, well below the 5% reroll threshold.

Acceptance target: essentially eliminate "remained high after rerolls" warnings;
generated sd(eta) distribution should overlap substantially with NUTS sd(eta_hat)
range (0.9–3.3), while still covering the full range of plausible Bernoulli signals.

## Guardrails

- Do not fit hyperpriors directly to the test set.
- Use real-NUTS posterior summaries only as sanity bounds for where real parameter
  mass appears, not as empirical Bayes targets.
- Keep generated data broad enough to cover plausible real datasets, not just the
  current benchmark pool.
