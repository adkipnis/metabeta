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

import math
from typing import NamedTuple

import torch
from torch import Tensor

from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import (
    hasSigmaEps,
    logLikelihood,
    logProbCorrRfx,
    logProbFfx,
    logProbSigma,
)
from metabeta.utils.regularization import unconstrainedToCholeskyCorr


_LOG_2PI = math.log(2 * math.pi)


class NCPParams(NamedTuple):
    ffx: Tensor                # (b, s, d)
    sigma_rfx: Tensor          # (b, s, q)
    sigma_eps: Tensor | None   # (b, s) — None for non-Normal
    u: Tensor                  # (b, m, s, q)
    z_corr: Tensor | None      # (b, s, d_corr) — None if independent rfx


class HierarchicalModel:
    """Differentiable joint log p(y, θ_g, u) for a padded batch of hierarchical datasets.

    Parameters
    ----------
    data : dict
        Padded batch tensors: X (b, m, n, d), Z (b, m, n, q), y (b, m, n),
        masks, and prior hyperparameters (nu_ffx, tau_ffx, tau_rfx, ...).
    likelihood_family : int
        0 = Normal, 1 = Bernoulli, 2 = Poisson.
    eps : float
        Numerical floor added to scale parameters.
    """

    def __init__(
        self,
        data: dict[str, Tensor],
        likelihood_family: int = 0,
        eps: float = 1e-12,
    ) -> None:
        self.likelihood_family = likelihood_family
        self.has_sigma_eps = hasSigmaEps(likelihood_family)
        self.eps = eps

        # Observations (unsqueeze last dim for broadcasting with (b, m, n, s))
        self.X = data['X']                         # (b, m, n, d)
        self.Z = data['Z']                         # (b, m, n, q)
        self.y = data['y'].unsqueeze(-1)           # (b, m, n, 1)
        self.mask_n = data['mask_n'].unsqueeze(-1) # (b, m, n, 1)

        # Raw masks — reshaped as needed per operation
        self._mask_m = data['mask_m']              # (b, m)
        self._mask_q = data['mask_q']              # (b, q)

        # Masks pre-shaped for logProbFfx / logProbSigma (expect (b, 1, d/q))
        self._mask_d_lp = data['mask_d'].unsqueeze(-2)    # (b, 1, d)
        self._mask_q_lp = data['mask_q'].unsqueeze(-2)    # (b, 1, q)
        self._mask_mq = data['mask_mq'].unsqueeze(-2)     # (b, m, 1, q)

        # Priors (unsqueeze sample dim for broadcasting with (b, s, d/q))
        self.nu_ffx = data['nu_ffx'].unsqueeze(-2)             # (b, 1, d)
        self.tau_ffx = data['tau_ffx'].unsqueeze(-2) + eps     # (b, 1, d)
        self.tau_rfx = data['tau_rfx'].unsqueeze(-2) + eps     # (b, 1, q)
        self.family_ffx = data['family_ffx']                   # (b,)
        self.family_sigma_rfx = data['family_sigma_rfx']       # (b,)
        if self.has_sigma_eps:
            self.tau_eps = data['tau_eps'].unsqueeze(-1) + eps # (b, 1)
            self.family_sigma_eps = data['family_sigma_eps']   # (b,)

        self.eta_rfx: Tensor | None = data.get('eta_rfx')     # (b,) or None
        self.q = data['mask_q'].shape[-1]                      # padded rfx dim

    # ------------------------------------------------------------------
    # Batch slicing
    # ------------------------------------------------------------------

    def sliceBatch(self, bi: int) -> 'HierarchicalModel':
        """Return a view of this model containing only batch element bi (b=1).

        Used by Hessian loops that need per-element calls to logJoint without
        cross-batch contamination of the boolean-indexed log-prob functions.
        """
        m = HierarchicalModel.__new__(HierarchicalModel)
        m.likelihood_family = self.likelihood_family
        m.has_sigma_eps = self.has_sigma_eps
        m.eps = self.eps
        m.q = self.q
        m.X = self.X[bi:bi + 1]
        m.Z = self.Z[bi:bi + 1]
        m.y = self.y[bi:bi + 1]
        m.mask_n = self.mask_n[bi:bi + 1]
        m._mask_m = self._mask_m[bi:bi + 1]
        m._mask_q = self._mask_q[bi:bi + 1]
        m._mask_d_lp = self._mask_d_lp[bi:bi + 1]
        m._mask_q_lp = self._mask_q_lp[bi:bi + 1]
        m._mask_mq = self._mask_mq[bi:bi + 1]
        m.nu_ffx = self.nu_ffx[bi:bi + 1]
        m.tau_ffx = self.tau_ffx[bi:bi + 1]
        m.tau_rfx = self.tau_rfx[bi:bi + 1]
        m.family_ffx = self.family_ffx[bi:bi + 1]
        m.family_sigma_rfx = self.family_sigma_rfx[bi:bi + 1]
        if self.has_sigma_eps:
            m.tau_eps = self.tau_eps[bi:bi + 1]
            m.family_sigma_eps = self.family_sigma_eps[bi:bi + 1]
        m.eta_rfx = self.eta_rfx[bi:bi + 1] if self.eta_rfx is not None else None
        return m

    # ------------------------------------------------------------------
    # NCP ↔ centered conversion
    # ------------------------------------------------------------------

    def rfxFromU(self, sigma_rfx: Tensor, u: Tensor, z_corr: Tensor | None = None) -> Tensor:
        """Compute centered rfx from non-centered offsets u.

        Independent (z_corr is None):
            rfx = u * σ_rfx              broadcasts (b, m, s, q) × (b, s, q)

        Correlated:
            L_full = diag(σ_rfx) @ L_corr    (b, s, q, q)
            rfx[b, m, s, :] = u[b, m, s, :] @ L_full[b, s, :, :].T

        Returns (b, m, s, q).
        """
        if z_corr is None:
            return u * sigma_rfx.unsqueeze(1)

        L_corr = unconstrainedToCholeskyCorr(z_corr, self.q)  # (b, s, q, q)
        # diag(sigma_rfx) @ L_corr: multiply row i of L_corr by sigma_rfx[b, s, i]
        L_full = sigma_rfx.unsqueeze(-1) * L_corr             # (b, s, q, q)
        # rfx[b,m,s,j] = sum_i u[b,m,s,i] * L_full[b,s,j,i]
        return torch.einsum('bmsi,bsji->bmsj', u, L_full)

    def uFromRfx(self, sigma_rfx: Tensor, rfx: Tensor, z_corr: Tensor | None = None) -> Tensor:
        """Inverse of rfxFromU: recover non-centered u from centered rfx.

        Returns (b, m, s, q).
        """
        if z_corr is None:
            return rfx / sigma_rfx.unsqueeze(1).clamp(min=self.eps)

        L_corr = unconstrainedToCholeskyCorr(z_corr, self.q)  # (b, s, q, q)
        L_full = sigma_rfx.unsqueeze(-1) * L_corr             # (b, s, q, q)

        # Solve L_full @ u_j = rfx_j  per group (lower-triangular)
        b, m, s, q = rfx.shape
        L_exp = L_full.unsqueeze(1).expand(b, m, s, q, q)     # (b, m, s, q, q)
        return torch.linalg.solve_triangular(
            L_exp, rfx.unsqueeze(-1), upper=False
        ).squeeze(-1)                                          # (b, m, s, q)

    # ------------------------------------------------------------------
    # Log joint
    # ------------------------------------------------------------------

    def logJoint(
        self,
        ffx: Tensor,                   # (b, s, d)
        sigma_rfx: Tensor,             # (b, s, q)
        sigma_eps: Tensor | None,      # (b, s) or None
        u: Tensor,                     # (b, m, s, q)
        z_corr: Tensor | None = None,  # (b, s, d_corr)
    ) -> Tensor:
        """Differentiable log p(y, θ_g, u) under NCP. Returns (b, s).

        log p = log p(y | ffx, σ_eps, rfx(u, σ_rfx))
              + Σ_j log p(u_j)         [N(0, I), summed over valid groups × rfx dims]
              + log p(ffx)
              + log p(σ_rfx)
              + log p(σ_eps)           [Normal only]
              + log p(z_corr)          [LKJ, if correlated]
        """
        rfx = self.rfxFromU(sigma_rfx, u, z_corr)    # (b, m, s, q)

        # Likelihood — pass zero sigma_eps for families that ignore it
        _sigma_eps = sigma_eps if sigma_eps is not None else ffx.new_zeros(ffx.shape[:2])
        ll = logLikelihood(
            ffx, _sigma_eps, rfx, self.y, self.X, self.Z, self.mask_n,
            likelihood_family=self.likelihood_family,
        )   # (b, s)

        # Global priors
        lp = logProbFfx(ffx, self.nu_ffx, self.tau_ffx, self.family_ffx, self._mask_d_lp)
        lp = lp + logProbSigma(
            sigma_rfx, self.tau_rfx, self.family_sigma_rfx, self._mask_q_lp
        )
        if self.has_sigma_eps and sigma_eps is not None:
            lp = lp + logProbSigma(sigma_eps, self.tau_eps, self.family_sigma_eps)

        # LKJ prior on correlation (if correlated rfx)
        if z_corr is not None and self.eta_rfx is not None:
            lp = lp + logProbCorrRfx(z_corr, self.q, self.eta_rfx)

        # Prior on u: N(0, I), masked over valid (group, rfx-dim) pairs
        # u: (b, m, s, q) — sum over dims 1 (m) and 3 (q) → (b, s)
        mask_u = (
            self._mask_m[:, :, None, None]       # (b, m, 1, 1)
            * self._mask_q[:, None, None, :]     # (b, 1, 1, q)
        )
        log_p_u = (-0.5 * (u.pow(2) + _LOG_2PI) * mask_u).sum(dim=(1, 3))
        lp = lp + log_p_u

        return ll + lp

    # ------------------------------------------------------------------
    # Proposal ↔ NCPParams conversion
    # ------------------------------------------------------------------

    def initFromProposal(self, proposal: Proposal) -> NCPParams:
        """Extract NCP parameters from a flow Proposal.

        Converts the centered rfx (samples_l) to non-centered u by solving
        u = rfx / σ_rfx (independent) or u = L_full⁻¹ rfx (correlated).
        Returns detached tensors; callers should clone and set requires_grad.
        """
        ffx = proposal.ffx.detach()               # (b, s, d)
        sigma_rfx = proposal.sigma_rfx.detach()   # (b, s, q)
        sigma_eps = proposal.sigma_eps.detach() if self.has_sigma_eps else None

        z_corr: Tensor | None = None
        if proposal.d_corr > 0:
            z_corr = proposal.samples_g[..., -proposal.d_corr :].detach()  # (b, s, d_corr)

        rfx = proposal.samples_l.detach()         # (b, m, s, q)
        u = self.uFromRfx(sigma_rfx, rfx, z_corr)

        return NCPParams(ffx=ffx, sigma_rfx=sigma_rfx, sigma_eps=sigma_eps, u=u, z_corr=z_corr)

    def toProposal(
        self,
        ffx: Tensor,                   # (b, s, d)
        sigma_rfx: Tensor,             # (b, s, q)
        sigma_eps: Tensor | None,      # (b, s) or None
        u: Tensor,                     # (b, m, s, q)
        z_corr: Tensor | None = None,  # (b, s, d_corr)
    ) -> Proposal:
        """Pack optimized NCP parameters into a Proposal.

        Converts u back to centered rfx, assembles samples_g / samples_l, and
        fills log_prob with zeros (no flow density available post-optimization).
        """
        with torch.no_grad():
            rfx = self.rfxFromU(sigma_rfx, u, z_corr)   # (b, m, s, q)

        b, s = ffx.shape[:2]
        m = rfx.shape[1]
        d_corr = z_corr.shape[-1] if z_corr is not None else 0

        parts: list[Tensor] = [ffx.detach(), sigma_rfx.detach()]
        if self.has_sigma_eps and sigma_eps is not None:
            parts.append(sigma_eps.detach().unsqueeze(-1))
        if z_corr is not None:
            parts.append(z_corr.detach())
        samples_g = torch.cat(parts, dim=-1)               # (b, s, D_g)

        proposed = {
            'global': {
                'samples': samples_g,
                'log_prob': samples_g.new_zeros(b, s),
            },
            'local': {
                'samples': rfx.detach(),                   # (b, m, s, q)
                'log_prob': rfx.new_zeros(b, m, s),
            },
        }
        return Proposal(proposed, has_sigma_eps=self.has_sigma_eps, d_corr=d_corr)
