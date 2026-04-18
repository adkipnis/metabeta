"""
posthoc/coordinate.py — Coordinate descent for flow posteriors.

Design
------
Alternates between two subproblems that are cheaper than joint optimization:

  θ_g-step  Optimize w.r.t. global params (ffx, log_σ_rfx, log_σ_eps, z_corr).
            - Normal: maximise the marginal log p(y|θ_g) + log p(θ_g), integrating
              out rfx analytically via the Woodbury identity.  This is the M-step of
              the EM algorithm for LMMs and is more stable than joint MAP — it avoids
              the σ_rfx/u coupling funnel.
            - GLMM: maximise log_joint conditioned on current u (Adam, n_g_steps).

  u-step    Update u given fixed θ_g.
            - Normal: E-step — set u to the exact conditional posterior mean
              E[u_j | θ_g, y] via Normal-Normal conjugacy.
            - GLMM: Adam steps on log p(y_j | θ_g, rfx_j(u_j)) + log p(u_j).

Output
------
For Normal, EM converges all s particles to the unique marginal MAP of θ_g.
The final u-step draws s independent samples from p(rfx | θ_g_MAP, y) (Gibbs
draw, not just the mean), giving correct conditional uncertainty for rfx.
θ_g has zero posterior variance in the output — it is a MAP point estimate.

For GLMM, different starting particles may retain diversity if the joint
posterior has multiple modes.

Notes
-----
- For Normal, coordinate descent converges monotonically in marginal log p(y, θ_g).
- For GLMM, convergence is not guaranteed monotone with approximate u-steps.
- A fresh Adam optimizer is created each outer cycle to avoid stale momentum.
"""
