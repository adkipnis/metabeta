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
    # Init-value construction
    # ------------------------------------------------------------------

    def _initVals(self, proposal: Proposal, b_idx: int) -> list[dict]:
        """Select n_chains diverse start points from proposal and back-transform.

        Diversity: samples at evenly-spaced quantiles of log_prob_g.
        Falls back to uniform spacing if log_prob_g is unavailable.
        """
        C = self.n_chains
        d, q, m = self.d, self.q, self.m

        sg = proposal.samples_g[b_idx].cpu()   # (n_s, D_g)
        sl = proposal.samples_l[b_idx].cpu()   # (m_batch, n_s, q_batch)
        n_s = sg.shape[0]

        # Diverse indices: quantiles of log density
        try:
            lp = proposal.log_prob_g[b_idx].cpu()   # (n_s,)
            sorted_idx = torch.argsort(lp)
            qs = torch.linspace(0, 1, C + 2)[1:-1]  # C interior quantiles
            pick = (qs * n_s).long().clamp(0, n_s - 1)
            indices = sorted_idx[pick].tolist()
        except (AttributeError, KeyError):
            indices = torch.linspace(0, n_s - 1, C).long().tolist()

        # The flow always outputs d_ffx fixed effects and d_rfx sigma values.
        # Use proposal.d / proposal.q (full layout) to locate sigma_rfx and
        # sigma_eps correctly; only extract the first d/q values for this dataset.
        p_d = proposal.d   # full d_ffx in the global tensor
        p_q = proposal.q   # full d_rfx in the global tensor
        ffx_s = sg[:, :d]                    # (n_s, d) — first d of d_ffx
        sigma_rfx_s = sg[:, p_d : p_d + q]  # (n_s, q) — first q of d_rfx, at correct offset

        # Cache corr_rfx for efficiency (computed lazily by Proposal)
        corr_rfx_all: Tensor | None = proposal.corr_rfx  # (b, n_s, q, q) or None

        init_list = []
        for s_idx in indices:
            iv: dict = {}

            # Fixed effects
            ffx_i = ffx_s[s_idx].numpy()
            for j in range(d):
                iv['Intercept' if j == 0 else f'x{j}'] = ffx_i[j]

            # sigma_eps (Normal only; PyMC applies log-transform internally)
            if self.has_sigma_eps:
                s_eps = float(sg[s_idx, p_d + p_q].item())  # at correct offset in flow tensor
                iv['sigma'] = max(s_eps, 1e-6)

            sr_i = sigma_rfx_s[s_idx].numpy().clip(1e-6)      # (q,)
            rfx_i = sl[:m, s_idx, :q].numpy()               # (m, q) — trim groups and rfx dims

            if self.correlated:
                # Build Cholesky of Σ_rfx = D @ R @ D
                if corr_rfx_all is not None:
                    R = corr_rfx_all[b_idx, s_idx, :q, :q].cpu().numpy()  # (q, q) — actual q
                else:
                    R = np.eye(q, dtype=np.float32)
                D_mat = np.diag(sr_i)
                Sigma = D_mat @ R @ D_mat + 1e-6 * np.eye(q)
                chol = np.linalg.cholesky(Sigma)   # lower triangular
                # rfx = z @ chol.T  →  z = solve(chol, rfx.T).T
                z = np.linalg.solve(chol, rfx_i.T).T   # (m, q)
                iv['_rfx_offset'] = z
                # sigma / corr parts of LKJCholeskyCov left at PyMC defaults

            else:
                for j in range(q):
                    s_name = '1|i_sigma' if j == 0 else f'x{j}|i_sigma'
                    o_name = '1|i_offset' if j == 0 else f'x{j}|i_offset'
                    iv[s_name] = float(sr_i[j])
                    iv[o_name] = rfx_i[:, j] / (sr_i[j] + 1e-12)   # (m,)

            init_list.append(iv)

        return init_list

    # ------------------------------------------------------------------
    # Trace → Proposal conversion
    # ------------------------------------------------------------------

    def _traceToProposal(self, trace: az.InferenceData) -> Proposal:
        """Convert a PyMC NUTS trace to a Proposal with b=1."""
        d, q = self.d, self.q
        out = extractAll(trace, self.ds, d, q, 'wn')

        # Shapes from extractAll:
        #   wn_ffx:       (d, n_s)
        #   wn_sigma_rfx: (q, n_s)
        #   wn_sigma_eps: (1, n_s)   — if Normal
        #   wn_rfx:       (q, m, n_s)
        #   wn_corr_rfx:  (1, n_s, q, q)

        # arviz returns float64 arrays; cast to float32 for compatibility with the
        # rest of the pipeline (model tensors, evaluation, etc.).
        def _f32(a) -> torch.Tensor:
            return torch.as_tensor(a).float()

        ffx = _f32(out['wn_ffx']).T                           # (n_s, d)
        sigma_rfx = _f32(out['wn_sigma_rfx']).T               # (n_s, q)
        n_s = ffx.shape[0]
        parts = [ffx, sigma_rfx]

        if self.has_sigma_eps:
            sigma_eps = _f32(out['wn_sigma_eps']).squeeze(0)  # (n_s,)
            parts.append(sigma_eps.unsqueeze(-1))              # (n_s, 1)

        samples_g = torch.cat(parts, dim=-1).unsqueeze(0)               # (1, n_s, D)
        # (q, m, n_s) → permute(1, 2, 0) → (m, n_s, q) → unsqueeze(0) → (1, m, n_s, q)
        samples_l = _f32(out['wn_rfx']).permute(1, 2, 0).unsqueeze(0)

        proposed = {
            'global': {
                'samples': samples_g,
                'log_prob': torch.zeros(1, n_s),           # dummy
            },
            'local': {
                'samples': samples_l,
                'log_prob': torch.zeros(1, self.m, n_s),   # dummy
            },
        }

        # extractAll always stores corr_rfx (identity for non-correlated datasets)
        corr_rfx = _f32(out['wn_corr_rfx'])   # (1, n_s, q, q)
        return Proposal(proposed, has_sigma_eps=self.has_sigma_eps, corr_rfx=corr_rfx)

    # ------------------------------------------------------------------
