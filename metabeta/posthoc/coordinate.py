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
from metabeta.utils.regularization import unconstrainedToCholesky


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
        ffx: Tensor,  # (b, s, d)  leaf
        log_sr: Tensor,  # (b, s, q)  leaf
        log_se: Tensor | None,  # (b, s)     leaf or None
        z_corr: Tensor | None,  # (b, s, d_corr) leaf or None
        u_fixed: Tensor,  # (b, m, s, q) detached — only used for GLMM
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
                    ffx,
                    sr,
                    se,
                    self.model.y,
                    self.model.X,
                    self.model.Z,
                    self.model.mask_n,
                    self.model._mask_m.unsqueeze(-1),
                )   # (b, s)
                lp = logProbFfx(
                    ffx,
                    self.model.nu_ffx,
                    self.model.tau_ffx,
                    self.model.family_ffx,
                    self.model._mask_d_lp,
                )
                lp = lp + logProbSigma(
                    sr,
                    self.model.tau_rfx,
                    self.model.family_sigma_rfx,
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
        ffx: Tensor,  # (b, s, d)
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
            L_corr = unconstrainedToCholesky(z_corr, q)  # (b, s, q, q)
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
            noise = (
                torch.linalg.solve_triangular(chol_M.mT, z.unsqueeze(-1), upper=True).squeeze(-1)
                * active
            )  # (b, m, s, q)
            rfx_out = rfx_mean + noise
        else:
            rfx_out = rfx_mean

        # Convert centered rfx to NCP u
        u = self.model.uFromRfx(sigma_rfx, rfx_out, z_corr)
        return u.detach()

    def _uStepGLMM(
        self,
        ffx: Tensor,  # (b, s, d) detached
        sigma_rfx: Tensor,  # (b, s, q) detached
        sigma_eps: Tensor | None,  # (b, s) detached or None
        z_corr: Tensor | None,
        u_current: Tensor,  # (b, m, s, q)
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
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, proposal: Proposal) -> tuple[Proposal, dict]:
        """Run coordinate cycles on all flow particles.

        Parameters
        ----------
        proposal : Proposal
            Flow proposal; all s samples used as independent starting particles.

        Returns
        -------
        proposal_out : Proposal
            Refined particles (same n_samples as input).
        diagnostics : dict
            'log_joint_curve' : list[float]  — mean log_joint per cycle
            'n_cycles'        : int          — cycles completed (may be < n_outer)
        """
        ncp = self.model.initFromProposal(proposal)

        # Leaf tensors for θ_g optimization
        ffx = ncp.ffx.detach().clone().requires_grad_(True)
        log_sr = ncp.sigma_rfx.clamp(min=1e-8).log().detach().clone().requires_grad_(True)
        log_se: Tensor | None = None
        if self.model.has_sigma_eps and ncp.sigma_eps is not None:
            log_se = ncp.sigma_eps.clamp(min=1e-8).log().detach().clone().requires_grad_(True)
        z_corr: Tensor | None = None
        if ncp.z_corr is not None:
            z_corr = ncp.z_corr.detach().clone().requires_grad_(True)

        # Local params: start from NCP-converted flow rfx; will be replaced each u-step
        u = ncp.u.detach().clone()

        log_joint_curve: list[float] = []

        for _ in range(self.n_outer):
            sr = log_sr.exp().detach()
            se = log_se.exp().detach() if log_se is not None else None
            zc = z_corr.detach() if z_corr is not None else None

            # u-step (before g-step so θ_g is consistent on first pass)
            if self.lf == 0:
                u = self._uStepNormal(ffx.detach(), sr, se, zc)
            else:
                u = self._uStepGLMM(ffx.detach(), sr, se, zc, u)

            # θ_g-step
            self._gStep(ffx, log_sr, log_se, z_corr, u)

            # Track convergence using log_joint (includes u for diagnostics)
            with torch.no_grad():
                sr_now = log_sr.exp()
                se_now = log_se.exp().detach() if log_se is not None else None
                zc_now = z_corr.detach() if z_corr is not None else None
                lj = self.model.logJoint(ffx.detach(), sr_now, se_now, u, zc_now)
                log_joint_curve.append(float(lj.mean().item()))

            if len(log_joint_curve) > 1:
                if abs(log_joint_curve[-1] - log_joint_curve[-2]) < self.tol:
                    break

        # Final u-step: sample from the conditional posterior (not the mean) to
        # give rfx proper uncertainty.  θ_g is the MAP (replicated s times);
        # rfx ~ p(rfx | θ_g_MAP, y) gives s independent Rao-Blackwellised draws.
        with torch.no_grad():
            sr_final = log_sr.exp().detach()
            se_final = log_se.exp().detach() if log_se is not None else None
            zc_final = z_corr.detach() if z_corr is not None else None
            if self.lf == 0:
                u = self._uStepNormal(ffx.detach(), sr_final, se_final, zc_final, sample=True)
            # For GLMM, u is already the latest Adam update — no change needed.

            proposal_out = self.model.toProposal(ffx.detach(), sr_final, se_final, u, zc_final)

        proposal_out.tpd = proposal.tpd
        return proposal_out, {
            'log_joint_curve': log_joint_curve,
            'n_cycles': len(log_joint_curve),
        }


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def runCoordinateDescent(
    model: Approximator,
    data: dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> tuple[Proposal, dict]:
    """Draw a flow proposal and refine with coordinate descent.

    cfg fields
    ----------
    n_samples         : int   — flow samples (= number of particles)
    cd_n_outer        : int   — coordinate cycles (default 20)
    cd_n_g_steps      : int   — Adam steps per g-step (default 30)
    cd_n_u_steps      : int   — Adam steps per u-step for GLMM (default 10)
    cd_lr             : float — Adam learning rate (default 0.05)
    cd_tol            : float — early-stopping tolerance (default 1e-4)
    rescale           : bool
    likelihood_family : int
    """
    lf = getattr(cfg, 'likelihood_family', 0)
    n_outer = getattr(cfg, 'cd_n_outer', 20)
    n_g_steps = getattr(cfg, 'cd_n_g_steps', 30)
    n_u_steps = getattr(cfg, 'cd_n_u_steps', 10)
    lr = getattr(cfg, 'cd_lr', 5e-2)
    tol = getattr(cfg, 'cd_tol', 1e-4)

    proposal = model.estimate(data, n_samples=cfg.n_samples)
    if cfg.rescale:
        proposal.rescale(data['sd_y'])
        data = rescaleData(data)

    gen_model = HierarchicalModel(data, likelihood_family=lf)
    cd = CoordinateDescent(
        gen_model,
        n_outer=n_outer,
        n_g_steps=n_g_steps,
        n_u_steps=n_u_steps,
        lr=lr,
        tol=tol,
    )
    return cd(proposal)
