"""
posthoc/svgd.py — Stein Variational Gradient Descent for flow posteriors.

Design
------
SVGD moves a set of s particles {θ_i} toward the target posterior by combining
an attractive force (gradient of log p) with a repulsive force (RBF kernel
gradient) that prevents collapse to the MAP:

    φ(θ_i) = (1/s) Σ_j [ k(θ_j, θ_i) ∇_{θ_j} log p(θ_j|y)
                          + ∇_{θ_j} k(θ_j, θ_i) ]
    θ_i ← θ_i + lr · φ(θ_i)

Parameterisation
----------------
Global params are packed into unconstrained (b, s, D_g):
    [ffx (d) | log_σ_rfx (q) | log_σ_eps (1, Normal only) | z_corr (d_corr)]

Normal likelihood (lf=0) — Rao-Blackwellised
    SVGD operates only on global params using the marginal log-likelihood
    (rfx integrated out analytically).  After convergence, rfx is drawn once
    from the exact conditional p(rfx | θ_g, y) — the same strategy as
    coordinate.py's final u-step.  This avoids the high-dimensional u space
    and the σ_rfx/u coupling funnel.

GLMM (lf>0)
    SVGD operates jointly on global (b, s, D_g) and local u (b, m, s, q)
    using the full log_joint.  Local particles are flattened to (b, s, m*q)
    for kernel computation.

Bandwidth
---------
Per-dimension median heuristic: h_d = median(off-diagonal (θ_i_d − θ_j_d)²) / (2 ln s).
Using a separate h_d per parameter dimension prevents tight dimensions (e.g. σ_eps)
from being over-repelled relative to broad ones (e.g. ffx).
Recomputed every step from the current particle cloud.

Notes
-----
- Particles are initialized from the flow proposal (already a good warm start).
- Gradient clipping (element-wise) guards against large early-step gradients.
- SVGD is asymptotically correct as s → ∞ and lr → 0.  With s=100–500 flow
  particles as initialization, convergence in 50–200 steps is typical.
"""

import argparse
import math

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


class SVGDRefiner:
    """SVGD post-hoc refinement for flow posteriors.

    Parameters
    ----------
    model : HierarchicalModel
    n_steps : int
        Number of SVGD update steps.
    lr : float
        Step size (equivalent to Adam lr; typical range 1e-3 – 5e-2).
    grad_clip : float
        Element-wise gradient clamp applied before kernel weighting.
    """

    def __init__(
        self,
        model: HierarchicalModel,
        n_steps: int = 100,
        lr: float = 1e-2,
        grad_clip: float = 10.0,
        lr_decay: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        lr_decay : float
            Final lr as a fraction of initial lr, using a cosine schedule.
            1.0 = constant (default).  0.1 = cosine decay to lr/10.
        """
        self.model = model
        self.n_steps = n_steps
        self.lr = lr
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lf = model.likelihood_family

