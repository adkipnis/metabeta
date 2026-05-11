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

## Guardrails

- Do not fit hyperpriors directly to the test set.
- Use real-NUTS posterior summaries only as sanity bounds for where real parameter
  mass appears, not as empirical Bayes targets.
- Keep Bernoulli and Poisson priors unchanged until their outcome-shape diagnostics
  are reviewed separately.
- Keep generated data broad enough to cover plausible real datasets, not just the
  current benchmark pool.
