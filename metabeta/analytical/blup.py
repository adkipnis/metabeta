"""Analytical Gaussian BLUP context utilities."""

import torch

from metabeta.utils.regularization import unconstrainedToCholesky


def analyticalBLUPContext(
    data: dict[str, torch.Tensor],
    beta: torch.Tensor,  # (B, S, d) fixed effects — constrained scale
    sigma_rfx: torch.Tensor,  # (B, S, q) — constrained scale
    sigma_eps: torch.Tensor,  # (B, S)   — constrained scale
    z_corr: torch.Tensor | None,  # (B, S, d_corr) unconstrained atanh or None
    clamp: float = 20.0,
) -> torch.Tensor:
    """Analytical BLUP mean, marginal std, and shrinkage given global parameter samples.

    Uses the closed-form Gaussian posterior (family == 0 only).
    Non-normal GLMMs would need per-sample Newton/Laplace solves for every group
    rather than a closed form, so they currently use the fixed PQL BLUP stats.
    Returns (B, m, S, 3*q).
    """
    from metabeta.posthoc.gaussian_local import analyticalBLUPStats

    q = sigma_rfx.shape[-1]
    d = beta.shape[-1]

    Sigma_rfx_inv: torch.Tensor | None = None
    if z_corr is not None:
        L_corr = unconstrainedToCholesky(z_corr, q)                    # (B, S, q, q)
        sr_inv_diag = torch.diag_embed(1.0 / sigma_rfx.clamp(min=1e-6))  # (B, S, q, q)
        A = torch.linalg.solve_triangular(L_corr, sr_inv_diag, upper=False)
        Sigma_rfx_inv = A.mT @ A                                        # (B, S, q, q)

    mu, blup_std, lambda_g = analyticalBLUPStats(
        data['y'],
        data['X'][..., :d],
        data['Z'][..., :q],
        beta,
        sigma_rfx,
        sigma_eps,
        data['mask_n'],
        Sigma_rfx_inv=Sigma_rfx_inv,
    )
    return torch.cat([mu.clamp(-clamp, clamp), blup_std.clamp(max=clamp), lambda_g], dim=-1)
