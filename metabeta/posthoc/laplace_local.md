# Conditional Laplace hybrid for Bernoulli GLMMs

## Context

`gaussianHybrid` (`metabeta/posthoc/gaussian_local.py:268`) replaces the local NPE samples with draws from the exact Gaussian–Gaussian conjugate conditional

$$b_i \mid y, \theta_g^{(s)} \;\sim\; \mathcal{N}\!\left(\Lambda^{-1} Z^\top(y - X\beta)/\sigma_\varepsilon^2,\;\Lambda^{-1}\right),\quad \Lambda = Z^\top Z/\sigma_\varepsilon^2 + \Sigma_{rfx}^{-1}.$$

It empirically improves LOO-NLL over the learned local flow for `likelihood_family == 0`. For Bernoulli (1) there is no closed-form conjugate, so the inference path in `approximator.py:584-606` falls back to the local NPE flow conditional on each global sample.

### Why gaussianHybrid wins — refined

The "double stochasticity" framing is partially correct but mis-localizes the gain:

- Both `gaussianHybrid` and the flow path do MC marginalization over the **same** $S$ samples from the (approximate) global posterior $q_\phi(\theta_g \mid y)$. The "outer" stochasticity is shared.
- The local NPE path additionally samples one draw from $q_\phi(b_i \mid y, \theta_g^{(s)})$ per global sample, while `gaussianHybrid` samples from the **exact** Gaussian conditional. Both have a stochastic inner draw — the difference is that the flow's conditional density carries amortization / capacity / training-vs-inference distribution-shift error, while the Gaussian conditional is closed-form and exact.
- Net effect: the marginal mixture $\int q(\cdot \mid y, \theta_g) q_\phi(\theta_g \mid y)\,\mathrm{d}\theta_g$ has zero conditional approximation error for the hybrid and non-zero error for the flow. This is what improves LOO-NLL, not "extra" stochasticity per se.

So the real lever is replacing the amortized conditional with an exact (or very accurate) one. The same lever exists for Bernoulli if we accept a fast approximation.

### Why this is feasible for Bernoulli — and why it ultimately fails

The conditional $p(b_i \mid y_i, \theta_g)$ for a Bernoulli GLMM with logit link is log-concave and well-approximated by a Gaussian centered at the conditional MAP (Laplace approximation / Fisher scoring). This is what `lme4::glmer` (PQL, AGQ) and INLA (nested Laplace) use internally. The $O(n_i^{-1})$ accuracy argument holds asymptotically, but in practice the Bernoulli RFX posterior is skewed — especially at small-to-moderate $n_i$ — and a Gaussian mode approximation misrepresents the shape. The trained local flow has implicitly learned this non-Gaussian shape from many examples and outperforms Laplace empirically (see Empirical results below).

## Approach: per-group, per-global-sample Laplace

For each $(b, m, s)$ triple solve
$$\hat b_{i,s} = \arg\max_{b}\; \log p(y_i \mid \beta_s, b) + \log p(b \mid \sigma_{rfx,s}, R_s)$$
via Newton–Raphson / Fisher scoring (canonical logit link makes observed = expected Hessian, both negative definite), then sample from $\mathcal{N}(\hat b_{i,s}, -H_{i,s}^{-1})$.

Per Newton step (vectorised over `(B, m, S)`):
- linear predictor `eta = X β + Z b` (we hold $\beta$ fixed; only the $Z b$ part changes between steps)
- mean: `mu = sigmoid(eta)`
- IRLS weights: `W = mu * (1 - mu)`
- gradient: `g = Z^T (y - mu) - Σ_rfx^{-1} b`
- Hessian (negative definite): `H = -(Z^T diag(W) Z + Σ_rfx^{-1})`
- update: `b ← b + (-H)^{-1} g` via Cholesky

Initialise `b = 0` (prior mean). Run a fixed budget of 10 Newton steps — vectorised over the full `(B, m, S)` grid, branching/masking is more expensive than the wasted iterations. Add `_adaptiveDiagonalJitter` for numerical safety in `-H`.

Sampling and log-density use the same Cholesky pattern as `analyticalRFX:115-122`:
```
L = cholesky(-H + jitter)
z ~ N(0, I_q)
b = b_hat + solve_triangular(L^T, z, upper=True)
log_prob = log_det_L - 0.5 * ||z||^2 - 0.5 * q * log(2π)
```

### Cost

`O(B · m · S · n_newton · (n_i · q + q²))`. Concrete: B=16, m=30, S=500, n_i=20, q=2, n_newton=10 → ~6·10⁸ cheap ops on GPU, comfortably under 500 ms per batch. Parallelises perfectly across `(B, m, S)`.

### Scope

Inference-only, mirroring `gaussianHybrid`. The training-time analytical BLUP injection (`approximator.py:_analyticalBLUPContext`, line 372) is **not** extended in this change — it is gated on `likelihood_family == 0` and stays that way. The hybrid swaps only the sampling-time conditional, not the training context.

This first pass targets `likelihood_family == 1` (Bernoulli) only. Poisson follows the same skeleton with `mu = exp(eta)` and `W = mu`, and is left for a follow-up once the Bernoulli path is validated.

## Files

### New: `metabeta/posthoc/laplace_local.py`

Public surface mirroring `gaussian_local.py`:

- `laplaceRFX(Y, X, Z, beta, sigma_rfx, mask_n, *, Sigma_rfx_inv=None, n_newton=10) -> (samples, log_prob)`
  - Shapes match `analyticalRFX` (`gaussian_local.py:73-122`): `samples (B,m,S,q)`, `log_prob (B,m,S)`.
  - Inner Newton loop, vectorised over `(B, m, S)`, fixed 10-step budget from `b = 0`.
  - Returns `log_prob` as the Gaussian density of the drawn `b` at the Laplace mode (analog to how `analyticalRFX` returns the conditional log density).
- `bernoulliHybrid(global_proposal, batch) -> Proposal`
  - Replaces the `local` part of an existing global `Proposal` with Laplace samples; preserves the global samples and their `log_prob` unchanged. Raises if `batch['likelihood_family'] != 1`.
- `bernoulliCeiling(batch, d_ffx, d_rfx, n_samples) -> Proposal`
  - Analog of `gaussianCeiling` (`gaussian_local.py:206`): global part is `n_samples` copies of the true globals (`log_prob_g = 0`); local part drawn from the Laplace approximation conditional on those true globals. Useful as a noise-ceiling reference for benchmarking how much LOO-NLL gap is closed by removing global-posterior error vs by removing Laplace approximation error.

### Modified: `metabeta/models/approximator.py`

- `analytical_local_posterior` property (lines 80–82): extend the gate from `likelihood_family == 0` to `likelihood_family in (0, 1)` while leaving family 2 on the flow path.
- `backward` method (lines 584–595): currently calls `gaussianHybrid`. Dispatch:
  - `family == 0` → `gaussianHybrid` (unchanged)
  - `family == 1` → `bernoulliHybrid`
- The zero-tensor placeholder for `proposed['local']` before `_postprocess` (lines 588–591) works for both branches.

### Reused (do not duplicate)

- `_correlationPrecision` (`gaussian_local.py:35`) for $\Sigma_{rfx}^{-1}$ when `corr_rfx` is present — identical math.
- `_adaptiveDiagonalJitter` (`gaussian_local.py:29`).
- `_safeSolve` from `metabeta.analytical.linalg`.
- `Proposal` wrapper (`metabeta.utils.evaluation`).
- `HierarchicalModel.logJoint` (`posthoc/generative.py:199`) as the oracle in tests — its gradient at the converged `b` should be ≈0.

## Verification

1. **Newton convergence (unit)**: synthesize a small Bernoulli batch (B=2, m=4, S=8, n_i=20, q=2), run `laplaceRFX`, assert max gradient norm < 1e-4 within 10 steps. ✓
2. **Oracle agreement (unit)**: for a synthetic batch, evaluate `HierarchicalModel.logJoint` w.r.t. `u` reparameterised from the Laplace mode `b̂`; gradient should be ≈0 and Hessian eigenvalues all negative. ✓
3. **Brute-force quadrature (q=1)**: for a Bernoulli toy batch with `q=1`, compare Laplace marginal log-likelihood per group against a 256-point trapezoidal integration of $\int p(y_i \mid b)\,p(b)\,\mathrm{d}b$ — relative error should be < 2% when $n_i \geq 5$. ✓
4. **End-to-end LOO-NLL**: run `experiments/posthoc/bernoulli_local.py` on a trained checkpoint. Result below.
5. **Timing**: not benchmarked separately; per-batch wall time was ~8 s on CPU for B=8, S=512, m≈30, q=2.

## Empirical results — Bernoulli, medium-b-mixed seed=15, valid, n=256

Checkpoint: `data=medium-b-mixed_model=large_seed=15`, 512 samples, 256 validation datasets.  
NUTS reference (same dataset): NRMSE=0.513, LOO-NLL=0.451.

```
         Method    NRMSE    NRMSE_rfx       R     ECE    LOO-NLL
---------------  -------  -----------  ------  ------  ---------
MB (flow local)   0.4981       0.7082  0.8081  0.0229     0.4666
     LapLoc(MB)   0.6059       1.3187  0.7588  0.0228     0.4873
   LapLoc(true)   0.2166       1.2272  0.9189  0.2257     0.4496
```

### Interpretation

**The Laplace local approximation hurts for Bernoulli.** Unlike the Gaussian case, replacing the flow's local posterior with Laplace increases both NRMSE (+22%) and LOO-NLL (+4.5%). This holds even for `LapLoc(true)`, which uses the *true* global parameters: NRMSE_rfx=1.23 vs the flow's 0.71, showing the Laplace approximation itself is the problem, not global posterior error.

The trained local flow outperforms Laplace because:
- Binary outcomes produce skewed, non-Gaussian RFX posteriors that Laplace (a Gaussian centered at the mode) misrepresents.
- The flow has been trained on many examples and has learned the true posterior shape, including non-Gaussian features of the Bernoulli likelihood.

**LapLoc(true) as noise ceiling**: removing all global posterior error collapses overall NRMSE to 0.22 (from 0.50), confirming that the global flow is the dominant bottleneck for parameter recovery. LOO-NLL improves only marginally (0.467 → 0.450), suggesting predictive performance is near the irreducible limit once the global posterior is good.

**MB flow vs NUTS**: the MB flow (LOO-NLL=0.467, NRMSE=0.498) is within ~3% of the NUTS ceiling (0.451 / 0.513) on both metrics. The global NPE posterior is **not** the bottleneck: on every NRMSE sub-metric MB matches or slightly beats NUTS (FFX: 0.288 vs 0.291, Σ(RFX): 0.502 vs 0.509, RFX: 0.709 vs 0.707). The residual LOO-NLL gap therefore reflects **distributional quality** rather than point-estimate accuracy — likely a combination of slightly miscalibrated posterior spread/shape (the higher Pareto k: 0.224 vs 0.164 signals worse IS efficiency) and the local flow's approximate conditional not perfectly capturing uncertainty around the correct conditional mean.

### What to try instead of Laplace

Laplace is the wrong approximation for the Bernoulli local posterior due to posterior asymmetry. Better alternatives, roughly in order of expected accuracy:

- **Gauss-Hermite (GH) quadrature**: for small $q$ (1–2), numerically integrate out $b_i$ on a fixed grid; exact up to quadrature error. Already in use for `sigma_rfx` refinement (`nAGQ`, see `metabeta/posthoc/gaussian_local.py`). Extension to $q > 2$ is expensive ($k^q$ nodes).
- **Variational Bayes (mean-field or full-covariance)**: maximise a Gaussian ELBO w.r.t. $(b, \Sigma_b)$; captures asymmetry better than Laplace if the ELBO geometry is well-behaved, at similar cost.
- **Improved training of the local flow**: the flow-based local posterior is already competitive with NUTS and outperforms Laplace. Capacity/training improvements (more flow steps, better conditioning) may close the residual gap more directly.

## Notes

- **Newton fallback**: if Newton diverges (e.g., highly separated Bernoulli group), cap at `n_newton=10` steps and accept the current iterate. Divergence is rare for the logit-link GLMM with a proper Gaussian prior on $b$; add Levenberg-style damping later only if it surfaces in practice.
- **Poisson follow-up**: the same module can absorb a `poissonHybrid` and `poissonCeiling` by swapping `mu` and `W` and clipping `eta` to avoid `exp` overflow (see `POISSON_ETA_CLIP_MAX` in `metabeta/utils/families.py`). Given the Bernoulli result, expect a similar story: the trained flow likely beats Laplace for Poisson too.
