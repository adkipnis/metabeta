"""
posthoc/laplace_glmm.py — Laplace-approximated Rao-Blackwellisation for non-Normal GLMMs.

Non-Normal likelihoods have no conjugate rfx conditional and no closed-form marginal
likelihood, so the exact Rao-Blackwellised SNIS of posthoc/importance.py (marginal=True)
does not apply. This module provides the Laplace analog, fully vectorized over
(datasets b, groups m, posterior samples s):

1. `laplaceRfxModes` — damped-Newton per-group conditional modes b*_j and Hessians
   H_j = ZᵀWZ + Σ⁻¹ of p(rfx_j | θ_g, y_j), warm-started at the flow's rfx draws
   (already close, so few iterations suffice).
2. `sampleRfxLaplace` — rfx ~ N(b*_j, H_j⁻¹): the Laplace conditional redraw, the
   non-conjugate analog of families.sampleRfxConditionalNormal.
3. `logMarginalLikelihoodLaplace` — log p̂(y_j | θ_g) = ℓ_j(b*) + log N(b*; 0, Σ_rfx)
   + (q/2)·log 2π − ½·log det H_j, the Laplace-approximated integrated likelihood
   (the same approximation lme4 uses at nAGQ=1). The two (q/2)·log 2π terms cancel,
   and padded rfx dims cancel between log det Σ and log det H exactly as in
   families.logMarginalLikelihoodNormal.
4. `LaplaceImportanceSampler` — SNIS with Laplace marginal weights + Laplace
   conditional rfx redraw. With attach_only=True the weights stay uniform and only
   the flow's rfx are replaced by conditional draws: zero weight bias (cannot fix
   global-parameter miscalibration, but directly targets local calibration — the
   biggest regression of the 'global'/'joint' IMH modes, see metropolis.py Findings).

The Laplace weights target the Laplace-approximated marginal posterior, not the exact
one: the bias is O(per-group Laplace error) — small for moderate group sizes, worst for
tiny Bernoulli groups — but crucially independent of flow quality, unlike the IMH
failure modes documented in metropolis.py.

For likelihood_family=0 (Normal) the Laplace approximation is exact; that path exists
to test this machinery end-to-end against logMarginalLikelihoodNormal.

Findings (2026-07 posthoc ablation, 128 validation datasets, small models)
--------------------------------------------------------------------------
Weight health is excellent (PSIS k ≈ 0.03 Bernoulli / 0.18 Poisson, 2–3% guardrail
fallback, ~69% sample efficiency), i.e. the flow is a good proposal for the Laplace
target — but the Laplace target itself is measurably biased: σ_rfx ECE shifts from
≈ +0.03/−0.01 (raw) to ≈ −0.10 on both families, the classic downward Laplace/PQL
bias for binary/count data with small groups tilting the σ_rfx posterior low. Net,
isLaplace is *not* better-calibrated than raw flow samples on these families;
attach_only ≈ raw (mild RFX-joint gains on Poisson, mild LOO-NLL loss on Bernoulli).

TODO
----
- nAGQ upgrade: for q ≤ 2, replace the Laplace integrated likelihood with adaptive
  Gauss-Hermite quadrature centred at (b*, H⁻¹) (reuse the _ghProductGrid pattern in
  analytical/glmm/bernoulli.py, vectorized over s) — directly targets the σ_rfx
  downward bias above at ~K× the likelihood-pass cost (K = grid size).
"""

import torch
from torch import Tensor
from torch import distributions as D
from torch.nn import functional as F

from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.families import POISSON_ETA_CLIP_MAX
from metabeta.utils.results import Proposal


def _sigmaChol(sigma_rfx: Tensor, L_corr: Tensor | None) -> Tensor:
    """Cholesky factor of Σ_rfx = D L_corr L_corrᵀ D (or diag(σ²)). (b, s, q, q)."""
    s = sigma_rfx.clamp(min=1e-6)
    if L_corr is None:
        return torch.diag_embed(s)
    return s.unsqueeze(-1) * L_corr


def _meanWeightScore(
    eta: Tensor,  # (b, m, n, s)
    y: Tensor,  # (b, m, n, 1)
    sigma_eps: Tensor,  # (b, s)
    likelihood_family: int,
) -> tuple[Tensor, Tensor]:
    """GLM working quantities for the Newton step.

    Returns (score_res, w): score wrt η is Zᵀ(score_res), Hessian weight is w —
    i.e. score_res = (y − μ)/φ and w = V(μ)/φ with dispersion φ and variance V.
    """
    if likelihood_family == 0:  # Normal: φ = σ²_eps, V = 1
        phi_inv = (1.0 / sigma_eps.pow(2).clamp(min=1e-12))[:, None, None, :]
        return (y - eta) * phi_inv, phi_inv.expand_as(eta)
    if likelihood_family == 1:  # Bernoulli: φ = 1, V = μ(1−μ)
        mu = torch.sigmoid(eta)
        return y - mu, (mu * (1.0 - mu)).clamp(min=1e-6)
    if likelihood_family == 2:  # Poisson: φ = 1, V = μ
        mu = torch.exp(eta.clamp(max=POISSON_ETA_CLIP_MAX))
        return y - mu, mu.clamp(min=1e-6)
    raise NotImplementedError(f'likelihood_family={likelihood_family}')


def _llPerGroup(
    eta: Tensor,  # (b, m, n, s)
    y: Tensor,  # (b, m, n, 1)
    sigma_eps: Tensor,  # (b, s)
    mask_n: Tensor,  # (b, m, n, 1)
    likelihood_family: int,
) -> Tensor:
    """Conditional log-likelihood summed within each group. Returns (b, m, s)."""
    if likelihood_family == 0:
        scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
        ll = D.Normal(loc=eta, scale=scale).log_prob(y)
    elif likelihood_family == 1:
        ll = y * eta - F.softplus(eta)
    elif likelihood_family == 2:
        eta_c = eta.clamp(max=POISSON_ETA_CLIP_MAX)
        ll = y * eta_c - torch.exp(eta_c) - torch.lgamma(y + 1.0)
    else:
        raise NotImplementedError(f'likelihood_family={likelihood_family}')
    return (ll * mask_n).sum(dim=2)  # (b, m, s)


def laplaceRfxModes(
    ffx: Tensor,  # (b, s, d)
    sigma_rfx: Tensor,  # (b, s, q)
    sigma_eps: Tensor,  # (b, s); ignored unless likelihood_family == 0
    y: Tensor,  # (b, m, n, 1)
    X: Tensor,  # (b, m, n, d)
    Z: Tensor,  # (b, m, n, q)
    mask_n: Tensor,  # (b, m, n, 1)
    mask_m: Tensor,  # (b, m, 1)
    likelihood_family: int,
    L_corr: Tensor | None = None,  # (b, s, q, q)
    init: Tensor | None = None,  # (b, m, s, q) warm start (e.g. flow rfx)
    n_newton: int = 5,
    damping: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-group conditional modes and Hessians of p(rfx_j | θ_g, y_j).

    Returns (modes, chol_H, Sigma_inv, L_rfx):
        modes    (b, m, s, q)     — Newton solution b*_j
        chol_H   (b, m, s, q, q)  — Cholesky of H_j = ZᵀW(b*)Z + Σ⁻¹
        Sigma_inv (b, s, q, q), L_rfx (b, s, q, q) — reusable Σ_rfx factors
    """
    b, s, q = sigma_rfx.shape
    m = X.shape[1]

    L_rfx = _sigmaChol(sigma_rfx, L_corr)
    eye = torch.eye(q, dtype=L_rfx.dtype, device=L_rfx.device)
    Sigma_inv = torch.cholesky_solve(eye.expand_as(L_rfx), L_rfx)  # (b, s, q, q)

    Z_m = Z * mask_n  # zero out padded observations
    mu_ffx = torch.einsum('bmnd,bsd->bmns', X, ffx)  # (b, m, n, s)
    mask_mq = mask_m.unsqueeze(-1)  # (b, m, 1, 1)

    modes = init.clone() if init is not None else y.new_zeros(b, m, s, q)
    modes = modes * mask_mq

    def hessian(w: Tensor) -> Tensor:
        ZWZ = torch.einsum('bmns,bmnq,bmnr->bmsqr', w * mask_n, Z_m, Z_m)
        return ZWZ + Sigma_inv.unsqueeze(1)

    for _ in range(n_newton):
        eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, modes)
        score_res, w = _meanWeightScore(eta, y, sigma_eps, likelihood_family)
        score = torch.einsum('bmnq,bmns->bmsq', Z_m, score_res * mask_n)
        score = score - torch.einsum('bsqr,bmsr->bmsq', Sigma_inv, modes)
        chol_H = torch.linalg.cholesky(hessian(w) + 1e-6 * eye)
        delta = torch.cholesky_solve(score.unsqueeze(-1), chol_H).squeeze(-1)
        modes = (modes + damping * delta) * mask_mq
        modes = modes.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    # final Hessian at the converged mode
    eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, modes)
    _, w = _meanWeightScore(eta, y, sigma_eps, likelihood_family)
    chol_H = torch.linalg.cholesky(hessian(w) + 1e-6 * eye)
    return modes, chol_H, Sigma_inv, L_rfx


def sampleRfxLaplace(modes: Tensor, chol_H: Tensor, mask_m: Tensor) -> Tensor:
    """Draw rfx ~ N(b*_j, H_j⁻¹) per group. Returns (b, m, s, q).

    chol_H is Chol(H) = Chol(V⁻¹), so Chol(V) = chol_H⁻ᵀ (upper-triangular solve).
    """
    z = torch.randn_like(modes)
    centered = torch.linalg.solve_triangular(chol_H.mT, z.unsqueeze(-1), upper=True).squeeze(-1)
    return (modes + centered) * mask_m.unsqueeze(-1)


def logMarginalLikelihoodLaplace(
    ffx: Tensor,  # (b, s, d)
    sigma_rfx: Tensor,  # (b, s, q)
    sigma_eps: Tensor,  # (b, s)
    y: Tensor,  # (b, m, n, 1)
    X: Tensor,  # (b, m, n, d)
    Z: Tensor,  # (b, m, n, q)
    mask_n: Tensor,  # (b, m, n, 1)
    mask_m: Tensor,  # (b, m, 1)
    likelihood_family: int,
    L_corr: Tensor | None = None,
    init: Tensor | None = None,
    n_newton: int = 5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Laplace-approximated marginal log-likelihood Σ_j log p̂(y_j | θ_g).

    log p̂(y_j|θ_g) = ℓ_j(b*) + log N(b*; 0, Σ) + (q/2) log 2π − ½ log det H_j
                   = ℓ_j(b*) − ½ (log det Σ + b*ᵀ Σ⁻¹ b* + log det H_j)

    (the two (q/2)·log 2π terms cancel; padded rfx dims cancel between the two
    log-dets exactly as in logMarginalLikelihoodNormal). Exact for Normal.

    Returns (ll (b, s), modes, chol_H) — modes/chol_H reusable for the
    conditional redraw so the weights and rfx draws share one target.
    """
    modes, chol_H, Sigma_inv, L_rfx = laplaceRfxModes(
        ffx,
        sigma_rfx,
        sigma_eps,
        y,
        X,
        Z,
        mask_n,
        mask_m,
        likelihood_family,
        L_corr=L_corr,
        init=init,
        n_newton=n_newton,
    )
    Z_m = Z * mask_n
    eta = torch.einsum('bmnd,bsd->bmns', X, ffx) + torch.einsum('bmnq,bmsq->bmns', Z_m, modes)
    ll_g = _llPerGroup(eta, y, sigma_eps, mask_n, likelihood_family)  # (b, m, s)

    diag_L = L_rfx.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
    log_det_Sigma = 2.0 * diag_L.log().sum(-1)  # (b, s)
    quad = torch.einsum('bmsq,bsqr,bmsr->bms', modes, Sigma_inv, modes)
    log_det_H = 2.0 * chol_H.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (b, m, s)

    laplace_g = ll_g - 0.5 * (log_det_Sigma[:, None, :] + quad + log_det_H)
    return (laplace_g * mask_m).sum(dim=1), modes, chol_H  # (b, s)


class LaplaceImportanceSampler(ImportanceSampler):
    """SNIS with Laplace marginal weights and Laplace conditional rfx redraw.

    The non-Normal analog of ImportanceSampler(marginal=True, rb_redraw=True):
    weights use the Laplace-approximated integrated likelihood, and rfx are
    redrawn from the Gaussian Laplace conditional N(b*, H⁻¹) at each global
    sample. With attach_only=True the weights are discarded (uniform) and only
    the rfx replacement is kept — zero weight bias at the cost of leaving global
    parameters uncorrected.
    """

    def __init__(
        self,
        data: dict[str, Tensor],
        attach_only: bool = False,
        n_newton: int = 5,
        **kwargs,
    ) -> None:
        if kwargs.get('marginal') or kwargs.get('full'):
            raise ValueError('LaplaceImportanceSampler defines its own marginal weights')
        rb_redraw = kwargs.pop('rb_redraw', True)
        super().__init__(data, **kwargs)
        self.rb_redraw = rb_redraw  # bypass parent's marginal-only validation
        self.attach_only = attach_only
        self.n_newton = n_newton
        self._modes: Tensor | None = None
        self._chol_H: Tensor | None = None

    def unnormalizedPosterior(self, proposal: Proposal) -> tuple[Tensor, Tensor]:
        lp, ffx, sigma_eps = self._logPriorGlobals(proposal)
        ll, modes, chol_H = logMarginalLikelihoodLaplace(
            ffx,
            proposal.sigma_rfx,
            sigma_eps,
            self.y,
            self.X,
            self.Z,
            self.mask_n,
            self.mask_m,
            self.likelihood_family,
            L_corr=self._getLCorr(proposal),
            init=proposal.rfx,
            n_newton=self.n_newton,
        )
        self._modes, self._chol_H = modes, chol_H
        return ll, lp

    def _redrawRfx(self, proposal: Proposal) -> None:
        rfx = sampleRfxLaplace(self._modes, self._chol_H, self.mask_m)
        proposal.data['local']['samples'] = rfx
        proposal.data['local']['log_prob'] = torch.zeros_like(proposal.log_prob_l)

    def __call__(self, proposal: Proposal) -> Proposal:
        proposal = super().__call__(proposal)
        if self.attach_only:
            # keep the diagnostics but drop the (approximate-target) weights:
            # downstream evaluation treats missing 'weights' as uniform
            proposal.is_results = {k: v for k, v in proposal.is_results.items() if k != 'weights'}
        return proposal
