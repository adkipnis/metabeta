"""
posthoc/laplace.py — MAP + Laplace approximation for flow posteriors.

Design
------
Uses HierarchicalModel.logJoint (NCP) as the optimization target.

Step 1 — MAP: Adam on all params in unconstrained space.
    ffx, u, z_corr  are unconstrained.
    sigma_rfx, sigma_eps are optimized in log-space and exponentiated.
    Initialization: highest log_prob_g sample from the flow proposal.

Step 2 — Hessian at MAP:
    Global params (ffx, log_σ_rfx, log_σ_eps, z_corr) — full (D_g × D_g)
    Hessian via two rounds of autograd.  D_g is small (d + q + 1 + d_corr ≤ ~15).

    Local params u — block-diagonal (q × q) per group, computed independently
    per (batch, group) pair.  Groups are conditionally independent given θ_g,
    so off-diagonal blocks are exactly zero.

Step 3 — Sampling:
    Global: eigendecompose −H_g; clamp negative/zero eigenvalues with jitter
    for numerical safety; draw n_samples from N(θ_g*, Σ_g) in unconstrained
    space; back-transform sigma params via exp().

    Local: Cholesky of −H_bj per active group; draw u_j samples from
    N(u_j*, Σ_j); inactive groups (mask_m = 0) stay at MAP.

    rfx is reconstructed from (sigma_rfx_samples, u_samples) via rfxFromU.
"""

import argparse

import torch
from torch import Tensor

from metabeta.models.approximator import Approximator
from metabeta.posthoc.generative import HierarchicalModel, NCPParams
from metabeta.utils.evaluation import Proposal
from metabeta.utils.preprocessing import rescaleData


class LaplaceRefiner:
    """MAP + Laplace approximation as a posthoc correction for flow posteriors.

    Parameters
    ----------
    model : HierarchicalModel
    n_steps : int
        Adam steps for MAP finding.
    lr : float
        Adam learning rate.
    n_samples : int
        Posterior samples to draw from the Laplace Gaussian.
    jitter : float
        Added to Hessian diagonal before inversion for numerical stability.
    """

    def __init__(
        self,
        model: HierarchicalModel,
        n_steps: int = 300,
        lr: float = 5e-2,
        n_samples: int = 1000,
        jitter: float = 1e-4,
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.lr = lr
        self.n_samples = n_samples
        self.jitter = jitter

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _bestInit(self, proposal: Proposal) -> NCPParams:
        """Pick the highest log_prob_g sample per batch element as MAP init."""
        b = proposal.samples_g.shape[0]
        best = proposal.log_prob_g.argmax(dim=-1)           # (b,)
        ba = torch.arange(b, device=proposal.samples_g.device)

        ffx = proposal.ffx[ba, best].unsqueeze(1)           # (b, 1, d)
        sigma_rfx = proposal.sigma_rfx[ba, best].unsqueeze(1)  # (b, 1, q)
        sigma_eps = None
        if self.model.has_sigma_eps:
            sigma_eps = proposal.sigma_eps[ba, best].unsqueeze(1)  # (b, 1)
        z_corr = None
        if proposal.d_corr > 0:
            z_corr = proposal.samples_g[ba, best, -proposal.d_corr:].unsqueeze(1)

        # Initialise u at zero (prior mean) rather than rfx/sigma_rfx.
        # This avoids large initial gradients when sigma_rfx is small.
        # The optimizer will move u to the posterior mode from here.
        b_sz, m = proposal.samples_l.shape[0], proposal.samples_l.shape[1]
        q = proposal.samples_l.shape[-1]
        u = torch.zeros(b_sz, m, 1, q, device=proposal.samples_g.device,
                        dtype=proposal.samples_g.dtype)
        return NCPParams(ffx=ffx, sigma_rfx=sigma_rfx, sigma_eps=sigma_eps, u=u, z_corr=z_corr)

    # ------------------------------------------------------------------
