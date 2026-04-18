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

    # ------------------------------------------------------------------
    # u-step: update local params
    # ------------------------------------------------------------------

    def _uStepNormal(
        self,
        ffx: Tensor,        # (b, s, d)
        sigma_rfx: Tensor,  # (b, s, q)
        sigma_eps: Tensor,  # (b, s)
        z_corr: Tensor | None,
        sample: bool = False,
    ) -> Tensor:
        """Posterior mean (or sample) of u | θ_g, y via Normal-Normal conjugacy.

        For each group j:
            M_j  = Σ_rfx⁻¹ + Z_j^T Z_j / σ²_eps           (q×q)
            μ_j  = M_j⁻¹ (Z_j^T r_j / σ²_eps)
            u_j  = uFromRfx(σ_rfx, μ_j)             if sample=False (EM E-step)
            u_j ~ N(μ_j, M_j⁻¹)                     if sample=True  (Gibbs draw)

        sample=False during EM cycles (clean convergence of θ_g).
        sample=True for the final output step (proper rfx uncertainty).

        Returns (b, m, s, q).
        """
        X = self.model.X          # (b, m, n, d)
        Z = self.model.Z          # (b, m, n, q)
        y = self.model.y          # (b, m, n, 1)
        mask_n = self.model.mask_n  # (b, m, n, 1)
        mask_m = self.model._mask_m  # (b, m)

        b, s = ffx.shape[:2]
        q = sigma_rfx.shape[-1]

        Z_m = Z * mask_n   # zero out padded observations
        ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_m, Z_m)   # (b, m, q, q)

        mu_ffx = torch.einsum('bmnd,bsd->bmns', X, ffx)
        r = (y - mu_ffx) * mask_n                           # (b, m, n, s)
        Ztr = torch.einsum('bmnq,bmns->bmsq', Z_m, r)      # (b, m, s, q)

        # Cholesky factor of Σ_rfx
        sigma_rfx_c = sigma_rfx.clamp(min=1e-6)
        if z_corr is not None:
            L_corr = unconstrainedToCholeskyCorr(z_corr, q)  # (b, s, q, q)
            L_rfx = L_corr * sigma_rfx_c.unsqueeze(-1)        # (b, s, q, q)
        else:
            L_rfx = torch.diag_embed(sigma_rfx_c)             # (b, s, q, q)

        # Σ_rfx⁻¹ via Cholesky solve
        eye = torch.eye(q, device=ffx.device, dtype=ffx.dtype).expand(b, s, q, q)
        Sigma_inv = torch.cholesky_solve(eye, L_rfx)   # (b, s, q, q)

        # M_j = Σ_rfx⁻¹ + Z_j^T Z_j / σ²_eps  (b, m, s, q, q)
        s2e = sigma_eps.pow(2)   # (b, s)
        M = Sigma_inv.unsqueeze(1) + ZtZ.unsqueeze(2) / s2e[:, None, :, None, None]

        jitter = 1e-6 * torch.eye(q, device=M.device, dtype=M.dtype)
        chol_M = torch.linalg.cholesky(M + jitter)   # (b, m, s, q, q)

        # Posterior mean: μ_j = M_j⁻¹ (Z_j^T r_j / σ²_eps)
        rhs = Ztr / s2e[:, None, :, None]            # (b, m, s, q)
        rfx_mean = torch.cholesky_solve(rhs.unsqueeze(-1), chol_M).squeeze(-1)
        # (b, m, s, q)

        active = mask_m[:, :, None, None].float()
        rfx_mean = rfx_mean * active

        if sample:
            # Sample from N(μ_j, M_j⁻¹): rfx = μ_j + chol_M⁻ᵀ z
            # chol_M is Chol(M) = Chol(V⁻¹), so Chol(V) = chol_M⁻ᵀ
            z = torch.randn_like(rfx_mean)
            noise = torch.linalg.solve_triangular(
                chol_M.mT, z.unsqueeze(-1), upper=True
            ).squeeze(-1) * active  # (b, m, s, q)
            rfx_out = rfx_mean + noise
        else:
            rfx_out = rfx_mean

        # Convert centered rfx to NCP u
        u = self.model.uFromRfx(sigma_rfx, rfx_out, z_corr)
        return u.detach()

    def _uStepGLMM(
        self,
        ffx: Tensor,               # (b, s, d) detached
        sigma_rfx: Tensor,         # (b, s, q) detached
        sigma_eps: Tensor | None,  # (b, s) detached or None
        z_corr: Tensor | None,
        u_current: Tensor,         # (b, m, s, q)
    ) -> Tensor:
        """Adam steps on u with θ_g frozen (GLMM).

        Maximises log p(y | θ_g, rfx(u)) + log p(u) in n_u_steps steps.
        Returns (b, m, s, q).
        """
        u = u_current.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=self.lr)
        for _ in range(self.n_u_steps):
            optimizer.zero_grad()
            lj = self.model.logJoint(ffx, sigma_rfx, sigma_eps, u, z_corr)
            loss = -lj.sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], max_norm=10.0)
            optimizer.step()
        return u.detach()

    # ------------------------------------------------------------------
