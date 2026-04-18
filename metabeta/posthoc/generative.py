"""
posthoc/generative.py — Differentiable PyTorch generative model for hierarchical regression.

Exposes the full joint log p(y, θ_g, u) in non-centered parameterization (NCP):

    rfx_j = u_j ⊙ σ_rfx                 (independent, z_corr is None)
    rfx_j = u_j @ L_full.T               (correlated, z_corr provided)

where L_full = diag(σ_rfx) @ L_corr and L_corr = choleskyCorr(z_corr).
u_j ~ N(0, I) for all groups j.

NCP vs centered
---------------
The centered parameterization couples rfx_j tightly to σ_rfx when σ_rfx is small,
creating funnel geometry that makes gradient-based optimization ill-conditioned.
NCP decouples them: u_j and σ_rfx have approximately independent gradients.
This is the same reparameterization that buildPymc uses for NUTS (see simulation/fit.py).

NPE posteriors are expressed in the centered parameterization (rfx as local flow
output).  init_from_proposal handles the one-shot transform to NCP.  to_proposal
inverts it after optimization.

Building block
--------------
HierarchicalModel.log_joint is a differentiable (b, s) map over leaf tensors.
Downstream methods (MAP, particle descent, coordinate descent) should:
  1. Call init_from_proposal to get NCPParams.
  2. Clone and set requires_grad_(True) on the tensors to optimize.
  3. Iterate: call log_joint, back-propagate, update.
  4. Call to_proposal to wrap the refined parameters.
"""

