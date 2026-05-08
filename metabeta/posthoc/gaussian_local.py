"""
posthoc/gaussian_local.py — Analytical Gaussian local posterior for LMMs (family == 0).

For y_i | β, b_i, σ_ε ~ N(X_i β + Z_i b_i, σ_ε² I) and
b_i | σ_rfx, R ~ N(0, D R D), where D = diag(σ_rfx):

    Λ_i = Z_i^T Z_i / σ_ε² + (D R D)^{-1}
    b_i | y_i, θ ~ N(Λ_i^{-1} Z_i^T (y_i - X_i β) / σ_ε², Λ_i^{-1})

Public API
----------
analyticalRFX   — sample from the exact local posterior, vectorised over B, m, S
gaussianCeiling — Proposal with analytical local | true globals (noise ceiling)
gaussianHybrid  — replace local samples of any Proposal with analytical ones
"""

import math

import torch
from torch import Tensor

from metabeta.analytical.glm import _safeSolve
from metabeta.utils.evaluation import Proposal
from metabeta.utils.regularization import corrToLower

_SIGMA_MIN = 1e-6


def _adaptiveDiagonalJitter(Lambda: Tensor, eps: float = 1e-6) -> Tensor:
    """Scale ridge per random-effect dimension, preserving padded-dim separation."""
    diag = Lambda.diagonal(dim1=-2, dim2=-1).abs().clamp(min=1.0)
    return torch.diag_embed(diag * eps)


def _correlationPrecision(
    sigma_rfx: Tensor,
    corr_rfx: Tensor | None = None,
    eta_rfx: Tensor | None = None,
) -> Tensor | None:
    """Build Σ_rfx^{-1} for optional correlated random effects.

    sigma_rfx: (B, S, q), corr_rfx: (B, q, q) or (B, S, q, q).
    eta_rfx == 0 marks independent-rfx datasets and forces identity correlation.
    """
    if corr_rfx is None:
        return None
    B, S, q = sigma_rfx.shape
    if q <= 1:
        return None

    corr = corr_rfx[..., :q, :q].to(device=sigma_rfx.device, dtype=sigma_rfx.dtype)
    if corr.dim() == 3:
        corr = corr.unsqueeze(1).expand(B, S, q, q)
    elif corr.dim() != 4:
        raise ValueError(f'corr_rfx has {corr.dim()} dims, expected 3 or 4')
    if corr.shape[:2] != (B, S):
        raise ValueError(
            f'corr_rfx leading shape {corr.shape[:2]} does not match sigma_rfx {(B, S)}'
        )

    eye = torch.eye(q, device=sigma_rfx.device, dtype=sigma_rfx.dtype)
    if eta_rfx is not None:
        active = eta_rfx.to(device=sigma_rfx.device).view(B, 1, 1, 1) > 0
        corr = torch.where(active, corr, eye.reshape(1, 1, q, q))

    jitter = 1e-6 * eye
    L_corr = torch.linalg.cholesky(corr + jitter)
    sr_inv_diag = torch.diag_embed(1.0 / sigma_rfx.clamp(min=_SIGMA_MIN))
    A = torch.linalg.solve_triangular(L_corr, sr_inv_diag, upper=False)
    return A.mT @ A


def analyticalRFX(
    Y: Tensor,
    X: Tensor,
    Z: Tensor,
    beta: Tensor,
    sigma_rfx: Tensor,
    sigma_eps: Tensor,
    mask_n: Tensor,
    Sigma_rfx_inv: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample from the exact Gaussian local posterior, one draw per global sample.

    Shapes: Y (B,m,n), X (B,m,n,d), Z (B,m,n,q), beta (B,S,d),
            sigma_rfx (B,S,q), sigma_eps (B,S), mask_n (B,m,n) bool.
    Sigma_rfx_inv: optional (B,S,q,q) full precision; if None uses diag(1/sigma_rfx^2).
    Returns: samples (B,m,S,q), log_prob (B,m,S).
    """
    B, m, _ = Y.shape
    S, q = beta.shape[1], Z.shape[-1]

    se = sigma_eps.clamp(min=_SIGMA_MIN)    # (B, S)
    sr = sigma_rfx.clamp(min=_SIGMA_MIN)    # (B, S, q)

    Z_m = Z * mask_n.float().unsqueeze(-1)  # zero-pad masked observations

    Xbeta = torch.einsum('bmnf,bsf->bmsn', X, beta)
    r = (Y.unsqueeze(2) - Xbeta) * mask_n.float().unsqueeze(2)  # (B, m, S, n)

    ZtZ = torch.einsum('bmnq,bmnp->bmpq', Z_m, Z_m)             # (B, m, q, q)
    ZtR = torch.einsum('bmnq,bmsn->bmsq', Z_m, r)               # (B, m, S, q)

    eps_sq = se**2
    prior_prec = (
        Sigma_rfx_inv.unsqueeze(1)
        if Sigma_rfx_inv is not None
        else torch.diag_embed(1.0 / sr**2).unsqueeze(1)
    )
    Lambda = ZtZ.unsqueeze(2) / eps_sq[:, None, :, None, None] + prior_prec

    Lam_j = Lambda + _adaptiveDiagonalJitter(Lambda)
    mu = _safeSolve(Lam_j, ZtR / eps_sq[:, None, :, None])  # (B, m, S, q)

    L = torch.linalg.cholesky(Lam_j)
    z = torch.randn(B, m, S, q, device=Y.device, dtype=Y.dtype)
    samples = mu + torch.linalg.solve_triangular(L.mT, z.unsqueeze(-1), upper=True).squeeze(-1)

    log_det_L = L.diagonal(dim1=-2, dim2=-1).clamp(min=1e-30).log().sum(-1)  # (B, m, S)
    log_prob = log_det_L - 0.5 * (z**2).sum(-1) - 0.5 * q * math.log(2.0 * math.pi)

    return samples, log_prob


def analyticalRFXMoments(
    Y: Tensor,
    X: Tensor,
    Z: Tensor,
    beta: Tensor,
    sigma_rfx: Tensor,
    sigma_eps: Tensor,
    mask_n: Tensor,
    Sigma_rfx_inv: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Analytical Gaussian local posterior mean and covariance.

    Shapes match analyticalRFX. Returns mean (B,m,S,q) and covariance
    (B,m,S,q,q), both conditioned on each global sample.
    """
    q = Z.shape[-1]

    se = sigma_eps.clamp(min=_SIGMA_MIN)
    sr = sigma_rfx.clamp(min=_SIGMA_MIN)

    Z_m = Z * mask_n.float().unsqueeze(-1)
    Xbeta = torch.einsum('bmnf,bsf->bmsn', X, beta)
    r = (Y.unsqueeze(2) - Xbeta) * mask_n.float().unsqueeze(2)

    ZtZ = torch.einsum('bmnq,bmnp->bmpq', Z_m, Z_m)
    ZtR = torch.einsum('bmnq,bmsn->bmsq', Z_m, r)

    eps_sq = se**2
    prior_prec = (
        Sigma_rfx_inv.unsqueeze(1)
        if Sigma_rfx_inv is not None
        else torch.diag_embed(1.0 / sr**2).unsqueeze(1)
    )
    Lambda = ZtZ.unsqueeze(2) / eps_sq[:, None, :, None, None] + prior_prec

    Lam_j = Lambda + _adaptiveDiagonalJitter(Lambda)

    mean = _safeSolve(Lam_j, ZtR / eps_sq[:, None, :, None])
    eye = torch.eye(q, device=Lam_j.device, dtype=Lam_j.dtype).reshape(
        (1,) * (Lam_j.dim() - 2) + (q, q)
    )
    covariance = _safeSolve(Lam_j, eye)
    return mean, covariance


def analyticalBLUPStats(
    Y: Tensor,
    X: Tensor,
    Z: Tensor,
    beta: Tensor,
    sigma_rfx: Tensor,
    sigma_eps: Tensor,
    mask_n: Tensor,
    Sigma_rfx_inv: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Analytical BLUP mean, marginal std, and shrinkage for the Gaussian rfx posterior.

    Shapes: Y (B,m,n), X (B,m,n,d), Z (B,m,n,q), beta (B,S,d),
            sigma_rfx (B,S,q), sigma_eps (B,S), mask_n (B,m,n) bool.
    Sigma_rfx_inv: optional (B,S,q,q) full precision; if None uses diag(1/sigma_rfx^2).
    Returns: mean (B,m,S,q), std (B,m,S,q), lambda_g (B,m,S,q).
    """
    sr = sigma_rfx.clamp(min=_SIGMA_MIN)
    mu, covariance = analyticalRFXMoments(
        Y,
        X,
        Z,
        beta,
        sigma_rfx,
        sigma_eps,
        mask_n,
        Sigma_rfx_inv=Sigma_rfx_inv,
    )
    Lambda_inv_diag = covariance.diagonal(dim1=-2, dim2=-1)  # (B, m, S, q)

    blup_std = Lambda_inv_diag.clamp(min=0.0).sqrt()
    lambda_g = (1.0 - Lambda_inv_diag / (sr.unsqueeze(1) ** 2 + 1e-8)).clamp(0.0, 1.0)

    return mu, blup_std, lambda_g


def gaussianCeiling(
    batch: dict[str, Tensor],
    d_ffx: int,
    d_rfx: int,
    n_samples: int = 512,
) -> Proposal:
    """Analytical local posterior conditioned on the TRUE global parameters.

    Returns a Proposal whose global part is n_samples copies of the true globals
    (log_prob_g = 0) and whose local part is drawn from the exact posterior.
    Only valid for likelihood_family == 0.
    """
    lf = batch.get('likelihood_family', 0)
    if hasattr(lf, 'item'):
        lf = lf.ravel()[0].item()
    if lf != 0:
        raise ValueError('gaussianCeiling is only valid for likelihood_family == 0')

    ffx = batch['ffx'][..., :d_ffx]           # (B, d_ffx)
    sigma_rfx = batch['sigma_rfx'][..., :d_rfx]     # (B, d_rfx)
    sigma_eps = batch['sigma_eps']                   # (B,)
    B = ffx.shape[0]

    beta_rep = ffx.unsqueeze(1).expand(-1, n_samples, -1)
    sr_rep = sigma_rfx.unsqueeze(1).expand(-1, n_samples, -1)
    se_rep = sigma_eps.unsqueeze(1).expand(-1, n_samples)
    corr_rfx = batch.get('corr_rfx')
    eta_rfx = batch.get('eta_rfx')
    Sigma_rfx_inv = _correlationPrecision(sr_rep, corr_rfx, eta_rfx)

    samples_l, log_prob_l = analyticalRFX(
        batch['y'],
        batch['X'][..., :d_ffx],
        batch['Z'][..., :d_rfx],
        beta_rep,
        sr_rep,
        se_rep,
        batch['mask_n'],
        Sigma_rfx_inv=Sigma_rfx_inv,
    )

    parts_g = [beta_rep, sr_rep, se_rep.unsqueeze(-1)]
    d_corr = d_rfx * (d_rfx - 1) // 2
    if d_corr > 0 and corr_rfx is not None:
        r_corr = corrToLower(corr_rfx[..., :d_rfx, :d_rfx])
        if eta_rfx is not None:
            r_corr = torch.where(eta_rfx.unsqueeze(-1) > 0, r_corr, torch.zeros_like(r_corr))
        parts_g.append(r_corr.unsqueeze(1).expand(-1, n_samples, -1))
    global_samples = torch.cat(parts_g, dim=-1)
    proposed = {
        'global': {
            'samples': global_samples,
            'log_prob': torch.zeros(B, n_samples, device=ffx.device),
        },
        'local': {
            'samples': samples_l,
            'log_prob': log_prob_l,
        },
    }
    return Proposal(proposed, has_sigma_eps=True, d_corr=d_corr if len(parts_g) == 4 else 0)


def gaussianHybrid(global_proposal: Proposal, batch: dict[str, Tensor]) -> Proposal:
    """Replace local samples of an existing Proposal with analytical Gaussian samples.

    The global part (samples and log_prob) is preserved unchanged.
    The local log_prob is the analytical conditional density for proposal accounting.
    Only valid for likelihood_family == 0.
    """
    lf = batch.get('likelihood_family', 0)
    if hasattr(lf, 'item'):
        lf = lf.ravel()[0].item()
    if lf != 0:
        raise ValueError('gaussianHybrid is only valid for likelihood_family == 0')

    q = global_proposal.q
    beta = global_proposal.ffx         # (B, S, d_ffx)
    sr = global_proposal.sigma_rfx   # (B, S, q)
    se = global_proposal.sigma_eps   # (B, S)
    Sigma_rfx_inv = _correlationPrecision(sr, global_proposal.corr_rfx, batch.get('eta_rfx'))

    samples_l, log_prob_l = analyticalRFX(
        batch['y'],
        batch['X'][..., : beta.shape[-1]],
        batch['Z'][..., :q],
        beta,
        sr,
        se,
        batch['mask_n'],
        Sigma_rfx_inv=Sigma_rfx_inv,
    )

    new_data = {
        'global': dict(global_proposal.data['global']),
        'local': {'samples': samples_l, 'log_prob': log_prob_l},
    }
    out = Proposal(
        new_data,
        has_sigma_eps=global_proposal.has_sigma_eps,
        d_corr=global_proposal.d_corr,
        corr_rfx=global_proposal.corr_rfx,
    )
    out.tpd = global_proposal.tpd
    return out
