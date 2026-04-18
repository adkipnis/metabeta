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

import argparse

import torch
from torch import Tensor

from metabeta.models.approximator import Approximator
from metabeta.posthoc.generative import HierarchicalModel
from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import (
    logMarginalLikelihoodNormal,
    logProbFfx,
    logProbSigma,
)
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.regularization import unconstrainedToCholeskyCorr


class CoordinateDescent:
    """Coordinate descent (EM-style) post-hoc refinement for flow posteriors.

    Parameters
    ----------
    model : HierarchicalModel
    n_outer : int
        Number of coordinate cycles (g-step + u-step).
    n_g_steps : int
        Adam steps per g-step per outer cycle.
    n_u_steps : int
        Adam steps per u-step for GLMM (ignored for Normal — exact update).
    lr : float
        Adam learning rate for both g- and u-steps.
    tol : float
        Early stopping: halt if |Δ mean log_joint| < tol between cycles.
    """

    def __init__(
        self,
        model: HierarchicalModel,
        n_outer: int = 20,
        n_g_steps: int = 30,
        n_u_steps: int = 10,
        lr: float = 5e-2,
        tol: float = 1e-4,
    ) -> None:
        self.model = model
        self.n_outer = n_outer
        self.n_g_steps = n_g_steps
        self.n_u_steps = n_u_steps
        self.lr = lr
        self.tol = tol
        self.lf = model.likelihood_family

    # ------------------------------------------------------------------
    # g-step: optimize global params
    # ------------------------------------------------------------------

    def _gStep(
        self,
        ffx: Tensor,             # (b, s, d)  leaf
        log_sr: Tensor,          # (b, s, q)  leaf
        log_se: Tensor | None,   # (b, s)     leaf or None
        z_corr: Tensor | None,   # (b, s, d_corr) leaf or None
        u_fixed: Tensor,         # (b, m, s, q) detached — only used for GLMM
    ) -> None:
        """In-place Adam update of global leaf tensors for n_g_steps iterations."""
        opt_params: list[Tensor] = [ffx, log_sr]
        if log_se is not None:
            opt_params.append(log_se)
        if z_corr is not None:
            opt_params.append(z_corr)

        optimizer = torch.optim.Adam(opt_params, lr=self.lr)

        for _ in range(self.n_g_steps):
            optimizer.zero_grad()
            sr = log_sr.exp()
            se = log_se.exp() if log_se is not None else None   # (b, s) or None

            if self.lf == 0:
                # Normal: marginal (integrates out rfx analytically)
                lml = logMarginalLikelihoodNormal(
                    ffx, sr, se,
                    self.model.y, self.model.X, self.model.Z,
                    self.model.mask_n, self.model._mask_m.unsqueeze(-1),
                )   # (b, s)
                lp = logProbFfx(
                    ffx, self.model.nu_ffx, self.model.tau_ffx,
                    self.model.family_ffx, self.model._mask_d_lp,
                )
                lp = lp + logProbSigma(
                    sr, self.model.tau_rfx, self.model.family_sigma_rfx,
                    self.model._mask_q_lp,
                )
                lp = lp + logProbSigma(se, self.model.tau_eps, self.model.family_sigma_eps)
                loss = -(lml + lp).sum()
            else:
                # GLMM: conditional on current u
                lj = self.model.logJoint(ffx, sr, se, u_fixed, z_corr)
                loss = -lj.sum()

            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=10.0)
            optimizer.step()

