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
