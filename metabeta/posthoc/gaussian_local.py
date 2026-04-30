"""
posthoc/gaussian_local.py — Analytical Gaussian local posterior for LMMs.

For likelihood_family == 0 (Gaussian linear mixed model):

    y_i | β, b_i, σ_ε ~ N(X_i β + Z_i b_i, σ_ε² I)
    b_i | σ_rfx       ~ N(0, diag(σ_rfx²))

the conditional posterior p(b_i | y_i, β, σ_ε, σ_rfx) is exactly Gaussian:

    Λ_i = Z_i^T Z_i / σ_ε² + diag(σ_rfx^{-2})
    μ_i = Λ_i^{-1} Z_i^T (y_i - X_i β) / σ_ε²
    b_i ~ N(μ_i, Λ_i^{-1})

Public API
----------
analyticalRFX     — core computation; vectorised over B, m, S
gaussianCeiling   — noise ceiling: analytical local | true globals
gaussianHybrid    — replace local samples of any Proposal with analytical ones
"""

import math

import torch
from torch import Tensor

from metabeta.utils.evaluation import Proposal

_SIGMA_MIN = 1e-6


def analyticalRFX(
    Y: Tensor,
    X: Tensor,
    Z: Tensor,
    beta: Tensor,
    sigma_rfx: Tensor,
    sigma_eps: Tensor,
    mask_n: Tensor,
) -> tuple[Tensor, Tensor]:
    """Analytical Gaussian posterior samples for b_i, vectorised over global samples.

    Args:
        Y         : (B, m, n_max)      — observations
        X         : (B, m, n_max, d)   — fixed-effects design matrix
        Z         : (B, m, n_max, q)   — random-effects design matrix
        beta      : (B, S, d)          — fixed-effects samples
        sigma_rfx : (B, S, q)          — random-effect SD samples
        sigma_eps : (B, S)             — observation noise SD samples
        mask_n    : (B, m, n_max) bool — valid-observation mask

    Returns:
        samples  : (B, m, S, q) — one RFX draw per global sample
        log_prob : (B, m, S)    — Gaussian log-density at the drawn sample
    """
    B, m, _ = Y.shape
    S, q = beta.shape[1], Z.shape[-1]

    sigma_eps_c = sigma_eps.clamp(min=_SIGMA_MIN)    # (B, S)
    sigma_rfx_c = sigma_rfx.clamp(min=_SIGMA_MIN)    # (B, S, q)

    # --- zero-out padded observations
    m_float = mask_n.float().unsqueeze(2)             # (B, m, 1, n_max)
    Z_m = Z * mask_n.float().unsqueeze(-1)            # (B, m, n_max, q)

    # --- residuals: y - X @ beta  →  (B, m, S, n_max)
    Xbeta = torch.einsum('bmnf,bsf->bmsn', X, beta)
    r = Y.unsqueeze(2) - Xbeta                        # (B, m, S, n_max)
    r = r * m_float                                    # mask padding

    # --- sufficient statistics (no S dependency for ZtZ)
    ZtZ = torch.einsum('bmnq,bmnp->bmpq', Z_m, Z_m)  # (B, m, q, q)
    ZtR = torch.einsum('bmnq,bmsn->bmsq', Z_m, r)     # (B, m, S, q)

    # --- precision matrix Λ = ZtZ/σ_ε² + diag(σ_rfx^{-2})  →  (B, m, S, q, q)
    eps_sq = sigma_eps_c**2                          # (B, S)
    ZtZ_scaled = ZtZ.unsqueeze(2) / eps_sq[:, None, :, None, None]
    prior_prec = torch.diag_embed(1.0 / sigma_rfx_c**2).unsqueeze(1)  # (B, 1, S, q, q)
    Lambda = ZtZ_scaled + prior_prec                   # (B, m, S, q, q)

    # --- posterior mean: solve Λ μ = ZtR / σ_ε²
    rhs = ZtR / eps_sq[:, None, :, None]               # (B, m, S, q)
    mu = torch.linalg.solve(Lambda, rhs)               # (B, m, S, q)

    # --- sample via Cholesky: x = μ + (L^T)^{-1} z,  z ~ N(0,I)
    L = torch.linalg.cholesky(Lambda)                  # (B, m, S, q, q) lower-triangular
    z = torch.randn(B, m, S, q, device=Y.device, dtype=Y.dtype)
    eps = torch.linalg.solve_triangular(L.mT, z.unsqueeze(-1), upper=True).squeeze(
        -1
    )                                       # (B, m, S, q)
    samples = mu + eps

    # --- log N(samples; μ, Λ^{-1})
    # = log|L| − ½‖L^T (x−μ)‖² − ½ q log 2π
    # Since x−μ = eps = (L^T)^{-1} z, we have L^T (x−μ) = z.
    log_det_L = L.diagonal(dim1=-2, dim2=-1).clamp(min=1e-30).log().sum(-1)  # (B, m, S)
    log_prob = log_det_L - 0.5 * (z**2).sum(-1) - 0.5 * q * math.log(2.0 * math.pi)

    return samples, log_prob


def gaussianCeiling(
    batch: dict[str, Tensor],
    d_ffx: int,
    d_rfx: int,
    n_samples: int = 512,
) -> Proposal:
    """Analytical local posterior conditioned on TRUE global parameters.

    The global part of the returned Proposal is a degenerate distribution
    (n_samples copies of the true parameters); log_prob_g is set to zeros.

    Only valid for likelihood_family == 0.

    Args:
        batch     : rescaled data batch (call rescaleData first)
        d_ffx     : model fixed-effects dimension (Approximator.d_ffx)
        d_rfx     : model random-effects dimension (Approximator.d_rfx)
        n_samples : local draws to take from the analytical posterior

    Returns:
        Proposal with shape (B, n_samples) global and (B, m, n_samples, q) local
    """
    lf = batch.get('likelihood_family', 0)
    if hasattr(lf, 'item'):
        lf = lf.ravel()[0].item()
    if lf != 0:
        raise ValueError('gaussianCeiling is only valid for likelihood_family == 0')

    ffx = batch['ffx'][..., :d_ffx]             # (B, d_ffx)
    sigma_rfx = batch['sigma_rfx'][..., :d_rfx]  # (B, d_rfx)
    sigma_eps = batch['sigma_eps']               # (B,)
    B = ffx.shape[0]

    # Degenerate global: n_samples copies of true params
    beta_rep = ffx.unsqueeze(1).expand(-1, n_samples, -1)           # (B, n_samples, d_ffx)
    sr_rep = sigma_rfx.unsqueeze(1).expand(-1, n_samples, -1)       # (B, n_samples, d_rfx)
    se_rep = sigma_eps.unsqueeze(1).expand(-1, n_samples)            # (B, n_samples)

    samples_l, _ = analyticalRFX(
        batch['y'],
        batch['X'][..., :d_ffx],
        batch['Z'][..., :d_rfx],
        beta_rep,
        sr_rep,
        se_rep,
        batch['mask_n'],
    )

    global_samples = torch.cat(
        [beta_rep, sr_rep, se_rep.unsqueeze(-1)], dim=-1
    )  # (B, n_samples, d_ffx + d_rfx + 1)

    proposed = {
        'global': {
            'samples': global_samples,
            'log_prob': torch.zeros(B, n_samples, device=ffx.device),
        },
        'local': {
            'samples': samples_l,
            'log_prob': torch.zeros(B, samples_l.shape[1], n_samples, device=ffx.device),
        },
    }
    return Proposal(proposed, has_sigma_eps=True)


def gaussianHybrid(
    global_proposal: Proposal,
    batch: dict[str, Tensor],
) -> Proposal:
    """Replace local samples of an existing Proposal with analytical Gaussian samples.

    The global part (samples and log_prob) is kept unchanged from global_proposal.
    The local part is replaced with one analytical draw per global sample.

    Only valid for likelihood_family == 0.

    Args:
        global_proposal : Proposal whose global samples provide β, σ_rfx, σ_ε
        batch           : rescaled data batch (call rescaleData first)

    Returns:
        Proposal with same global as input and analytical local samples
    """
    lf = batch.get('likelihood_family', 0)
    if hasattr(lf, 'item'):
        lf = lf.ravel()[0].item()
    if lf != 0:
        raise ValueError('gaussianHybrid is only valid for likelihood_family == 0')

    q = global_proposal.q
    beta = global_proposal.ffx                   # (B, S, d_ffx)
    sigma_rfx = global_proposal.sigma_rfx        # (B, S, q)
    sigma_eps = global_proposal.sigma_eps        # (B, S)

    d_ffx = beta.shape[-1]

    samples_l, log_prob_l = analyticalRFX(
        batch['y'],
        batch['X'][..., :d_ffx],
        batch['Z'][..., :q],
        beta,
        sigma_rfx,
        sigma_eps,
        batch['mask_n'],
    )

    new_data = {
        'global': dict(global_proposal.data['global']),
        'local': {'samples': samples_l, 'log_prob': log_prob_l},
    }
    out = Proposal(
        new_data,
        has_sigma_eps=global_proposal.has_sigma_eps,
        d_corr=global_proposal.d_corr,
    )
    out.tpd = global_proposal.tpd
    return out
