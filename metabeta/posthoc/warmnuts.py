"""
posthoc/warmnuts.py — Warm-started NUTS correction for flow posteriors.

Design
------
The flow q(θ) is used to generate n_chains diverse starting points for PyMC's
NUTS sampler.  Unlike importance sampling, NUTS targets the exact posterior
p(θ|y) without weights, and the flow start points (back-transformed to PyMC's
non-centred parameterisation) reduce the warm-up phase significantly.

Warm-start mechanics
--------------------
A single flow proposal is drawn for the full batch.  For each dataset at
batch index b, n_chains diverse samples are selected at quantiles of the
global log density.  Each sample is back-transformed:

  Independent rfx (eta_rfx == 0 or q == 1):
      z_j  = rfx_j / σ_rfx_j    →  '{1|i,x1|i,...}_offset'
      σ_rfx_j                    →  '{1|i,x1|i,...}_sigma'

  Correlated rfx (eta_rfx > 0, q >= 2):
      Σ_rfx = D @ R @ D  (D = diag(σ_rfx), R = corr_rfx from flow)
      chol  = lower_cholesky(Σ_rfx)
      z     = rfx @ chol⁻ᵀ     →  '_rfx_offset'
      (sigma/corr parts of LKJCholeskyCov left at PyMC defaults)

The resulting dict list is passed as `initvals` to pm.sample.

Output
------
WarmNuts.__call__ returns a Proposal with b=1 and n_chains * draws samples.
runWarmNuts stacks per-dataset proposals along the batch dimension.

Empirical comparison
--------------------
Compare WarmNuts proposals to flow-only and IMH proposals via
Evaluator.summary() / plotComparison.  Key diagnostics:
  - acceptance rate / R-hat (from trace, not yet propagated to Proposal)
  - NRMSE, coverage, SBC ranks via the standard evaluation pipeline
"""

