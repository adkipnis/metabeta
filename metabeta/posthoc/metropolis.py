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

import argparse
from typing import Literal

import torch
from torch import Tensor

from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import (
    hasSigmaEps,
    logMarginalLikelihoodNormal,
    logProbFfx,
    logProbSigma,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.regularization import unconstrainedToCholeskyCorr

Mode = Literal['global', 'marginal', 'joint']


class MetropolisSampler:
    def __init__(
        self,
        data: dict[str, Tensor],
        n_chains: int = 4,
        n_steps: int = 250,
        burnin: int = 50,
        mode: Mode = 'marginal',
        likelihood_family: int = 0,
        eps: float = 1e-12,
    ) -> None:
        if mode == 'marginal' and likelihood_family != 0:
            raise ValueError("mode='marginal' requires likelihood_family=0 (Normal)")
        if burnin >= n_steps:
            raise ValueError('burnin must be < n_steps')

        self.n_chains = n_chains
        self.n_steps = n_steps
        self.burnin = burnin
        self.mode = mode
        self.likelihood_family = likelihood_family
        self.has_sigma_eps = hasSigmaEps(likelihood_family)
        self.eps = eps

        # Reuse ImportanceSampler for prior log-prob computation and unnormalizedPosterior.
        # 'joint' needs full=True (includes rfx prior and local log-prob).
        self._is = ImportanceSampler(
            data, full=(mode == 'joint'), likelihood_family=likelihood_family, eps=eps
        )

        # Data tensors for the Normal-Normal conditional (marginal mode).
        # These mirror what ImportanceSampler stores but are kept as direct references.
        self._X = data['X']           # (b, m, n, d)
        self._Z = data['Z']           # (b, m, n, q)
        self._y = data['y'].unsqueeze(-1)  # (b, m, n, 1)
        self._mask_n = data['mask_n'].unsqueeze(-1)   # (b, m, n, 1)
        self._mask_m = data['mask_m'].unsqueeze(-1)   # (b, m, 1)

    # ------------------------------------------------------------------
    # Log-weight computation
    # ------------------------------------------------------------------

    def _logWeights(self, proposal: Proposal) -> Tensor:
        """Compute unnormalised log IS weights (b, s) according to self.mode."""
        log_q_g = proposal.log_prob_g  # (b, s)

        if self.mode == 'marginal':
            ffx = proposal.ffx         # (b, s, d)
            sigma_rfx = proposal.sigma_rfx   # (b, s, q)
            sigma_eps = proposal.sigma_eps   # (b, s)
            ll = logMarginalLikelihoodNormal(
                ffx, sigma_rfx, sigma_eps,
                self._y, self._X, self._Z, self._mask_n, self._mask_m,
            )
            lp = logProbFfx(ffx, self._is.nu_ffx, self._is.tau_ffx,
                            self._is.family_ffx, self._is.mask_d)
            lp = lp + logProbSigma(sigma_rfx, self._is.tau_rfx,
                                   self._is.family_sigma_rfx, self._is.mask_q)
            lp = lp + logProbSigma(sigma_eps, self._is.tau_eps,
                                   self._is.family_sigma_eps)
            return ll + lp - log_q_g

        # 'global' or 'joint': delegate to ImportanceSampler
        ll, lp = self._is.unnormalizedPosterior(proposal)
        if self.mode == 'joint':
            log_q_l = proposal.log_prob_l   # (b, m, s)
            lq = log_q_g + (log_q_l * self._is.mask_m).sum(1)
            return ll + lp - lq

        return ll + lp - log_q_g

    # ------------------------------------------------------------------
