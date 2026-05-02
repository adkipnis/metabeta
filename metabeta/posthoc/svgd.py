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
from metabeta.utils.regularization import unconstrainedToCholesky


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

    # ------------------------------------------------------------------
    # RBF kernel
    # ------------------------------------------------------------------

    def _rbfKernel(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Per-dimension RBF kernel with median-heuristic bandwidth.

        Uses a separate bandwidth h_d per dimension so that tight dimensions
        (e.g. σ_eps) are not over-repelled relative to broad ones (e.g. ffx):

            k(x, y) = exp(-Σ_d (x_d - y_d)² / h_d)

        X          : (b, s, D)
        Returns
        K          : (b, s, s)
        repulsion  : (b, s, D) — (1/s) Σ_j ∇_{X_i} k(X_i, X_j)
                                = (2/h_d) * (1/s) Σ_j K_ij (X_i - X_j)_d
        """
        b, s, D = X.shape
        diff = X.unsqueeze(2) - X.unsqueeze(1)   # (b, s, s, D)
        sq_per_dim = diff.pow(2)                   # (b, s, s, D)

        if s > 1:
            r, c = torch.triu_indices(s, s, offset=1, device=X.device)
            med = sq_per_dim[:, r, c, :].median(dim=1).values  # (b, D)
            h = (med / (2.0 * math.log(s))).clamp(min=1e-6)    # (b, D)
        else:
            h = X.new_ones(b, D)

        scaled = sq_per_dim / h[:, None, None, :]               # (b, s, s, D)
        K = torch.exp(-scaled.sum(-1))                           # (b, s, s)

        # repulsion[b, i, d] = (2/h_d) * (1/s) * Σ_j K[b,i,j] * diff[b,i,j,d]
        repulsion = (2.0 / (h[:, None, :] * s)) * torch.einsum(
            'bij,bijd->bid', K, diff
        )   # (b, s, D)

        return K, repulsion

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _marginalGrads(
        self,
        g: Tensor,   # (b, s, D_g) detached
        d: int,
        q: int,
        has_se: bool,
        has_zc: bool,
    ) -> tuple[Tensor, Tensor]:
        """∇ log p_marginal(θ_g | y) for all particles (Normal only).

        Returns (grad_g, log_p) both (b, s).
        """
        g_leaf = g.clone().requires_grad_(True)
        ffx = g_leaf[..., :d]
        sr = g_leaf[..., d:d + q].exp()
        cursor = d + q
        se = g_leaf[..., cursor].exp() if has_se else None
        if has_se:
            cursor += 1
        zc = g_leaf[..., cursor:] if has_zc else None

        lml = logMarginalLikelihoodNormal(
            ffx, sr, se,
            self.model.y, self.model.X, self.model.Z,
            self.model.mask_n, self.model._mask_m.unsqueeze(-1),
        )
        lp = logProbFfx(
            ffx, self.model.nu_ffx, self.model.tau_ffx,
            self.model.family_ffx, self.model._mask_d_lp,
        )
        lp = lp + logProbSigma(
            sr, self.model.tau_rfx, self.model.family_sigma_rfx, self.model._mask_q_lp,
        )
        lp = lp + logProbSigma(se, self.model.tau_eps, self.model.family_sigma_eps)
        log_p = lml + lp   # (b, s)
        log_p.sum().backward()

        grad_g = g_leaf.grad.clamp(-self.grad_clip, self.grad_clip).detach()
        return grad_g, log_p.detach()

    def _jointGrads(
        self,
        g: Tensor,   # (b, s, D_g) detached
        u: Tensor,   # (b, m, s, q) detached
        d: int,
        q: int,
        has_se: bool,
        has_zc: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """∇ log p(θ_g, u | y) for all particles (GLMM).

        Returns (grad_g, grad_u, log_p).
        """
        g_leaf = g.clone().requires_grad_(True)
        u_leaf = u.clone().requires_grad_(True)

        ffx = g_leaf[..., :d]
        sr = g_leaf[..., d:d + q].exp()
        cursor = d + q
        se = g_leaf[..., cursor].exp() if has_se else None
        if has_se:
            cursor += 1
        zc = g_leaf[..., cursor:] if has_zc else None

        log_p = self.model.logJoint(ffx, sr, se, u_leaf, zc)   # (b, s)
        log_p.sum().backward()

        grad_g = g_leaf.grad.clamp(-self.grad_clip, self.grad_clip).detach()
        grad_u = u_leaf.grad.clamp(-self.grad_clip, self.grad_clip).detach()
        return grad_g, grad_u, log_p.detach()

    # ------------------------------------------------------------------
    # SVGD steps
    # ------------------------------------------------------------------

    def _stepMarginal(
        self,
        g: Tensor,   # (b, s, D_g)
        d: int,
        q: int,
        has_se: bool,
        has_zc: bool,
        lr: float | None = None,
    ) -> tuple[Tensor, float]:
        """One SVGD step on global params using the marginal log p (Normal)."""
        if lr is None:
            lr = self.lr
        grad_g, log_p = self._marginalGrads(g, d, q, has_se, has_zc)
        K, rep = self._rbfKernel(g)
        phi = torch.bmm(K, grad_g) / g.shape[1] + rep   # (b, s, D_g)
        return g + lr * phi, float(log_p.mean().item())

    def _stepJoint(
        self,
        g: Tensor,   # (b, s, D_g)
        u: Tensor,   # (b, m, s, q)
        d: int,
        q: int,
        has_se: bool,
        has_zc: bool,
        lr: float | None = None,
    ) -> tuple[Tensor, Tensor, float]:
        """One SVGD step on global + local params using log_joint (GLMM)."""
        if lr is None:
            lr = self.lr
        b, s, D_g = g.shape
        m, q_u = u.shape[1], u.shape[-1]

        grad_g, grad_u, log_p = self._jointGrads(g, u, d, q, has_se, has_zc)

        K_g, rep_g = self._rbfKernel(g)
        phi_g = torch.bmm(K_g, grad_g) / s + rep_g

        u_flat = u.permute(0, 2, 1, 3).reshape(b, s, m * q_u)   # (b, s, m*q)
        K_u, rep_u = self._rbfKernel(u_flat)
        grad_u_flat = grad_u.permute(0, 2, 1, 3).reshape(b, s, m * q_u)
        phi_u_flat = torch.bmm(K_u, grad_u_flat) / s + rep_u
        phi_u = phi_u_flat.reshape(b, s, m, q_u).permute(0, 2, 1, 3)

        return g + lr * phi_g, u + lr * phi_u, float(log_p.mean().item())

    # ------------------------------------------------------------------
    # Final u-sampling (Normal only)
    # ------------------------------------------------------------------

    def _sampleU(
        self,
        ffx: Tensor,       # (b, s, d)
        sigma_rfx: Tensor, # (b, s, q)
        sigma_eps: Tensor, # (b, s)
        z_corr: Tensor | None,
    ) -> Tensor:
        """Draw rfx ~ p(rfx | θ_g, y) via Normal-Normal conjugacy; return u.

        For each group j:
            M_j  = Σ_rfx⁻¹ + Z_j^T Z_j / σ²_eps
            rfx_j ~ N(M_j⁻¹ Z_j^T r_j / σ²_eps,  M_j⁻¹)
        Returns u = uFromRfx(σ_rfx, rfx) of shape (b, m, s, q).
        """
        X, Z, y = self.model.X, self.model.Z, self.model.y
        mask_n, mask_m = self.model.mask_n, self.model._mask_m

        b, s = ffx.shape[:2]
        q = sigma_rfx.shape[-1]

        Z_m = Z * mask_n
        ZtZ = torch.einsum('bmnq,bmnr->bmqr', Z_m, Z_m)   # (b, m, q, q)

        mu_ffx = torch.einsum('bmnd,bsd->bmns', X, ffx)
        r = (y - mu_ffx) * mask_n                           # (b, m, n, s)
        Ztr = torch.einsum('bmnq,bmns->bmsq', Z_m, r)      # (b, m, s, q)

        sigma_rfx_c = sigma_rfx.clamp(min=1e-6)
        if z_corr is not None:
            L_corr = unconstrainedToCholesky(z_corr, q)
            L_rfx = L_corr * sigma_rfx_c.unsqueeze(-1)
        else:
            L_rfx = torch.diag_embed(sigma_rfx_c)

        eye = torch.eye(q, device=ffx.device, dtype=ffx.dtype).expand(b, s, q, q)
        Sigma_inv = torch.cholesky_solve(eye, L_rfx)

        s2e = sigma_eps.pow(2)
        M = Sigma_inv.unsqueeze(1) + ZtZ.unsqueeze(2) / s2e[:, None, :, None, None]
        jitter = 1e-6 * torch.eye(q, device=M.device, dtype=M.dtype)
        chol_M = torch.linalg.cholesky(M + jitter)

        rhs = Ztr / s2e[:, None, :, None]
        rfx_mean = torch.cholesky_solve(rhs.unsqueeze(-1), chol_M).squeeze(-1)

        active = mask_m[:, :, None, None].float()
        z = torch.randn_like(rfx_mean)
        noise = torch.linalg.solve_triangular(
            chol_M.mT, z.unsqueeze(-1), upper=True
        ).squeeze(-1) * active
        rfx = rfx_mean * active + noise

        return self.model.uFromRfx(sigma_rfx, rfx, z_corr)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, proposal: Proposal) -> tuple[Proposal, dict]:
        """Run SVGD on flow particles.

        Parameters
        ----------
        proposal : Proposal
            Flow proposal; all s samples used as starting particles.

        Returns
        -------
        proposal_out : Proposal
            Refined particles (same n_samples).
        diagnostics : dict
            'log_p_history' : list[float] — mean log p per step
        """
        ncp = self.model.initFromProposal(proposal)

        d = ncp.ffx.shape[-1]
        q = ncp.sigma_rfx.shape[-1]
        has_se = self.model.has_sigma_eps and ncp.sigma_eps is not None
        has_zc = ncp.z_corr is not None

        # Pack g: unconstrained global params (b, s, D_g)
        parts: list[Tensor] = [ncp.ffx, ncp.sigma_rfx.clamp(min=1e-8).log()]
        if has_se:
            parts.append(ncp.sigma_eps.clamp(min=1e-8).log().unsqueeze(-1))
        if has_zc:
            parts.append(ncp.z_corr)
        g = torch.cat(parts, dim=-1).detach()  # (b, s, D_g)

        u = ncp.u.detach()   # (b, m, s, q) — only used for GLMM

        log_p_history: list[float] = []
        denom = max(self.n_steps - 1, 1)

        for step in range(self.n_steps):
            # Cosine schedule: lr → lr * lr_decay over n_steps
            t = step / denom
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))
            lr_t = self.lr * (self.lr_decay + (1.0 - self.lr_decay) * cos_factor)
            if self.lf == 0:
                g, lp_val = self._stepMarginal(g, d, q, has_se, has_zc, lr=lr_t)
            else:
                g, u, lp_val = self._stepJoint(g, u, d, q, has_se, has_zc, lr=lr_t)
            log_p_history.append(lp_val)

        # Unpack final g
        with torch.no_grad():
            ffx_f = g[..., :d]
            sr_f = g[..., d:d + q].exp()
            cursor = d + q
            se_f: Tensor | None = None
            if has_se:
                se_f = g[..., cursor].exp()
                cursor += 1
            zc_f: Tensor | None = g[..., cursor:] if has_zc else None

            # Normal: sample rfx from exact conditional; GLMM: use final u
            if self.lf == 0:
                u = self._sampleU(ffx_f, sr_f, se_f, zc_f)

            proposal_out = self.model.toProposal(ffx_f, sr_f, se_f, u, zc_f)

        proposal_out.tpd = proposal.tpd
        return proposal_out, {'log_p_history': log_p_history}


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def runSVGD(
    model: Approximator,
    data: dict[str, torch.Tensor],
    cfg: argparse.Namespace,
) -> tuple[Proposal, dict]:
    """Draw a flow proposal and refine with SVGD.

    cfg fields
    ----------
    n_samples         : int   — flow samples (= n_particles)
    svgd_n_steps      : int   — SVGD steps (default 100)
    svgd_lr           : float — initial step size (default 0.01)
    svgd_grad_clip    : float — element-wise gradient clamp (default 10.0)
    svgd_lr_decay     : float — final lr fraction for cosine schedule (default 1.0)
    rescale           : bool
    likelihood_family : int
    """
    lf = getattr(cfg, 'likelihood_family', 0)
    n_steps = getattr(cfg, 'svgd_n_steps', 100)
    lr = getattr(cfg, 'svgd_lr', 1e-2)
    grad_clip = getattr(cfg, 'svgd_grad_clip', 10.0)
    lr_decay = getattr(cfg, 'svgd_lr_decay', 1.0)

    proposal = model.estimate(data, n_samples=cfg.n_samples)
    if cfg.rescale:
        proposal.rescale(data['sd_y'])
        data = rescaleData(data)

    gen_model = HierarchicalModel(data, likelihood_family=lf)
    refiner = SVGDRefiner(gen_model, n_steps=n_steps, lr=lr, grad_clip=grad_clip, lr_decay=lr_decay)
    return refiner(proposal)
