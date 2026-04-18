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
    # MAP finding
    # ------------------------------------------------------------------

    def _findMAP(self, init: NCPParams) -> tuple[NCPParams, Tensor, Tensor | None, Tensor | None, dict]:
        """Optimize logJoint to find MAP.

        Returns the MAP as NCPParams (s=1) together with the unconstrained
        log-sigma leaf tensors (log_sr, log_se) needed for Hessian computation
        in unconstrained space.
        """
        ffx = init.ffx.detach().clone().requires_grad_(True)
        log_sr = init.sigma_rfx.clamp(min=1e-8).log().detach().clone().requires_grad_(True)
        u = init.u.detach().clone().requires_grad_(True)

        opt_params: list[Tensor] = [ffx, log_sr, u]
        log_se: Tensor | None = None
        if self.model.has_sigma_eps and init.sigma_eps is not None:
            log_se = init.sigma_eps.clamp(min=1e-8).log().detach().clone().requires_grad_(True)
            opt_params.append(log_se)
        z_corr: Tensor | None = None
        if init.z_corr is not None:
            z_corr = init.z_corr.detach().clone().requires_grad_(True)
            opt_params.append(z_corr)

        optimizer = torch.optim.Adam(opt_params, lr=self.lr)
        losses: list[float] = []

        best_loss = float('inf')
        best_state = {p: p.detach().clone() for p in opt_params}

        for _ in range(self.n_steps):
            optimizer.zero_grad()
            lj = self.model.logJoint(
                ffx,
                log_sr.exp(),
                log_se.exp() if log_se is not None else None,
                u,
                z_corr,
            )
            loss = -lj.sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=10.0)
            optimizer.step()
            val = float(loss.detach())
            losses.append(val)
            if val < best_loss:
                best_loss = val
                best_state = {p: p.detach().clone() for p in opt_params}

        # Restore best observed params (guards against divergence near the end)
        with torch.no_grad():
            for p, v in best_state.items():
                p.copy_(v)

        with torch.no_grad():
            map_params = NCPParams(
                ffx=ffx.detach(),
                sigma_rfx=log_sr.exp().detach(),
                sigma_eps=log_se.exp().detach() if log_se is not None else None,
                u=u.detach(),
                z_corr=z_corr.detach() if z_corr is not None else None,
            )
        final = losses[-1] if losses else float('inf')
        return map_params, log_sr.detach(), log_se.detach() if log_se is not None else None, \
            z_corr.detach() if z_corr is not None else None, \
            {'loss_curve': losses, 'final_loss': final}

    # ------------------------------------------------------------------
    # Hessian computation
    # ------------------------------------------------------------------

    def _gFlat(
        self,
        ffx_b: Tensor,      # (1, d)
        log_sr_b: Tensor,   # (1, q)
        log_se_b: Tensor | None,  # (1,) or None
        zc_b: Tensor | None,      # (1, d_corr) or None
    ) -> Tensor:
        """Concatenate unconstrained global params into a flat (D_g,) vector."""
        parts: list[Tensor] = [ffx_b.squeeze(0), log_sr_b.squeeze(0)]
        if log_se_b is not None:
            parts.append(log_se_b.squeeze(0))
        if zc_b is not None:
            parts.append(zc_b.squeeze(0))
        return torch.cat(parts)

    def _ljFromGFlat(
        self,
        g: Tensor,              # (D_g,) unconstrained global vector for one batch elem
        u_b: Tensor,            # (1, m, 1, q) fixed
        d: int,
        q: int,
        has_se: bool,
        has_zc: bool,
        model: HierarchicalModel | None = None,  # sliced (b=1) model; defaults to self.model
    ) -> Tensor:
        """Scalar log_joint for a single batch element from a flat global vector."""
        model = model if model is not None else self.model
        cursor = 0
        ffx_b = g[cursor:cursor + d].reshape(1, 1, d)
        cursor += d
        sr_b = g[cursor:cursor + q].exp().reshape(1, 1, q)
        cursor += q
        se_b: Tensor | None = None
        if has_se:
            se_b = g[cursor:cursor + 1].exp().reshape(1, 1)  # (1, 1) = (b=1, s=1)
            cursor += 1
        zc_b: Tensor | None = None
        if has_zc:
            d_corr = g.shape[0] - cursor
            zc_b = g[cursor:].reshape(1, 1, d_corr)
        return model.logJoint(ffx_b, sr_b, se_b, u_b, zc_b).squeeze()

    def _globalHessian(
        self,
        ffx_map: Tensor,        # (b, 1, d)
        log_sr_map: Tensor,     # (b, 1, q)
        log_se_map: Tensor | None,  # (b, 1) or None
        zc_map: Tensor | None,  # (b, 1, d_corr) or None
        u_map: Tensor,          # (b, m, 1, q)
    ) -> Tensor:
        """Full (b, D_g, D_g) Hessian of logJoint w.r.t. concatenated unconstrained globals.

        Computed element-wise over the batch via two rounds of autograd.
        """
        b = ffx_map.shape[0]
        d, q = ffx_map.shape[-1], log_sr_map.shape[-1]
        has_se = log_se_map is not None
        has_zc = zc_map is not None
        D_g = d + q + (1 if has_se else 0) + (zc_map.shape[-1] if has_zc else 0)

        H = ffx_map.new_zeros(b, D_g, D_g)
        for bi in range(b):
            model_bi = self.model.sliceBatch(bi)
            log_se_bi = log_se_map[bi:bi + 1] if has_se else None
            zc_bi = zc_map[bi:bi + 1] if has_zc else None
            g0 = self._gFlat(ffx_map[bi], log_sr_map[bi], log_se_bi, zc_bi)
            g0 = g0.detach().requires_grad_(True)
            u_bi = u_map[bi:bi + 1]

            lj = self._ljFromGFlat(g0, u_bi, d, q, has_se, has_zc, model_bi)
            grad1 = torch.autograd.grad(lj, g0, create_graph=True)[0]  # (D_g,)
            H_bi = g0.new_zeros(D_g, D_g)
            for i in range(D_g):
                grad2 = torch.autograd.grad(
                    grad1[i], g0, retain_graph=(i < D_g - 1)
                )[0]
                H_bi[i] = grad2.detach()
            H[bi] = H_bi
        return H  # (b, D_g, D_g)

    def _localHessian(
        self,
        ffx_map: Tensor,        # (b, 1, d)
        log_sr_map: Tensor,     # (b, 1, q)
        log_se_map: Tensor | None,
        zc_map: Tensor | None,
        u_map: Tensor,          # (b, m, 1, q)
    ) -> Tensor:
        """Block-diagonal (b, m, q, q) Hessian of logJoint w.r.t. u, one block per group.

        Groups are conditionally independent given θ_g, so off-diagonal blocks
        are exactly zero.  Only active groups (mask_m == 1) are computed; the
        rest retain the identity (prior Hessian = -I).
        """
        b, m, _, q = u_map.shape
        d = ffx_map.shape[-1]
        has_se = log_se_map is not None
        has_zc = zc_map is not None
        H = u_map.new_zeros(b, m, q, q)

        for bi in range(b):
            model_bi = self.model.sliceBatch(bi)
            sr_bi = log_sr_map[bi].exp()                       # (1, q)
            se_bi = log_se_map[bi].exp() if has_se else None   # (1,) or None
            zc_bi = zc_map[bi:bi + 1] if has_zc else None
            n_active = int(self.model._mask_m[bi].sum().item())

            for j in range(n_active):
                u_j = u_map[bi, j, 0, :].detach().requires_grad_(True)  # (q,)

                # Rebuild u_full with u_j as the differentiable leaf
                u_full = torch.cat([
                    u_map[bi:bi + 1, :j].detach(),
                    u_j.reshape(1, 1, 1, q),
                    u_map[bi:bi + 1, j + 1:].detach(),
                ], dim=1)  # (1, m, 1, q)

                lj = model_bi.logJoint(
                    ffx_map[bi:bi + 1],
                    sr_bi.unsqueeze(0),
                    se_bi.unsqueeze(0) if se_bi is not None else None,
                    u_full,
                    zc_bi,
                ).squeeze()

                grad1 = torch.autograd.grad(lj, u_j, create_graph=True)[0]  # (q,)
                H_bj = u_j.new_zeros(q, q)
                for i in range(q):
                    grad2 = torch.autograd.grad(
                        grad1[i], u_j, retain_graph=(i < q - 1)
                    )[0]
                    H_bj[i] = grad2.detach()
                H[bi, j] = H_bj

            # Inactive groups: Hessian is -I (only the N(0,I) prior contributes)
            for j in range(n_active, m):
                H[bi, j] = -torch.eye(q, device=H.device, dtype=H.dtype)

        return H  # (b, m, q, q)

    # ------------------------------------------------------------------
