"""
posthoc/metropolis.py — Independence Metropolis-Hastings (IMH) correction for flow posteriors.

Design
------
The flow q(θ) approximates p(θ|y) but can be systematically off (e.g. the sigma_eps/sigma_rfx
ridge in Normal models).  IMH uses the flow as proposal in a Markov chain, accepting/rejecting
with ratio w(θ') / w(θ).  Unlike IS, degenerate proposals are corrected by rejection rather
than by collapsed weights, and the chain is guaranteed to target the correct posterior.

Three modes differ in how rfx (local params) are handled:

  'global'
      log_w = log p(y|θ_g, θ_l) + log p(θ_g) − log q_g
      rfx enter the likelihood but carry no IS correction.  When a global proposal is accepted
      the paired flow rfx from the same draw travels with it.  Works for all likelihoods.

  'marginal'  [default for Normal; requires likelihood_family == 0]
      log_w = log p_marginal(y|θ_g) + log p(θ_g) − log q_g
      rfx are integrated out analytically via the Normal-Normal conjugate marginal, so the
      chain operates only in the low-dimensional global space.  After acceptance, fresh rfx
      are drawn from the exact conditional posterior p(rfx | θ_g, y) (Rao-Blackwellised).
      Best mixing; theoretically optimal for Normal.

  'joint'
      log_w = log p(y|θ_g,θ_l) + log p(θ_l|θ_g) + log p(θ_g) − log q_g − log q_l
      Full joint target; rfx are subject to acceptance alongside globals.  Correct for GLMMs
      but degenerates with many groups (high rfx dimension).

Chain mechanics
---------------
A pool of s = n_chains × n_steps proposals is drawn upfront.  All log weights are computed in
one vectorised pass.  The MH loop is vectorised over (b, n_chains) and iterates as a Python
loop over n_steps — typically 50–250 steps.  The first `burnin` steps are discarded; the
remaining n_chains × (n_steps − burnin) samples are returned as a Proposal.

Empirical comparison
--------------------
Call MetropolisSampler with each mode in turn and evaluate the resulting Proposal via
Evaluator.summary() / plotComparison to choose the best mode for a given dataset type.
The diagnostics dict returned by __call__ contains per-chain acceptance rates as a quick
quality indicator before running the full evaluation.
"""

