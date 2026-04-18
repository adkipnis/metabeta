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

import argparse

import arviz as az
import numpy as np
import pymc as pm
import torch
from torch import Tensor

from metabeta.models.approximator import Approximator
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.simulation.fit import buildPymc, extractAll
from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import hasSigmaEps
from metabeta.utils.preprocessing import rescaleData


class WarmNuts:
    def __init__(
        self,
        ds: dict[str, np.ndarray],
        n_chains: int = 4,
        tune: int = 500,
        draws: int = 500,
        seed: int = 42,
        target_accept: float = 0.9,
    ) -> None:
        """
        Parameters
        ----------
        ds : dict
            Single unpadded dataset (output of Fitter._getSingle / unpad).
        n_chains : int
            Number of independent NUTS chains.
        tune : int
            NUTS tuning steps (burn-in; discarded by PyMC).
        draws : int
            Posterior draws per chain.
        seed : int
            Random seed passed to pm.sample.
        target_accept : float
            Target acceptance rate for step-size adaptation.  0.9 is
            recommended for posteriors with complex geometry (default 0.9).
        """
        self.ds = ds
        self.n_chains = n_chains
        self.tune = tune
        self.draws = draws
        self.seed = seed
        self.target_accept = target_accept

        self.d = int(ds['d'])
        self.q = int(ds['q'])
        self.m = int(ds['m'])
        self.correlated = float(ds.get('eta_rfx', 0)) > 0 and self.q >= 2
        self.has_sigma_eps = hasSigmaEps(int(ds.get('likelihood_family', 0)))

        # Build PyMC model once; reused across __call__ invocations.
        self.model = buildPymc(ds)

    # ------------------------------------------------------------------
