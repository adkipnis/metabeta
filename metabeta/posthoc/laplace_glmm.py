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

Findings (2026-09-12, 512 test datasets per run — see experiments/posthoc/LAPLACE_UPGRADES.md)
----------------------------------------------------------------------------------------------
Most of the apparent "Laplace σ_rfx bias" above was Newton-instability contamination
of the weights: full-step Newton oscillated on extreme proposals (worst on huge-count
Poisson), making log p̂(y|θ_g) warm-start-dependent by 1e4+ nats on samples near the
pool max weight. After robustifying laplaceRfxModes (backtracking line search,
adaptive iteration budget, 1-nat pinning guard), Poisson-large isLaplace σ_rfx ECE
went −0.046 → −0.031 (raw: −0.027) and Bernoulli-huge max PSIS k 15.4 → 2.6.
Post-fix, the AGQ weight upgrade (nagq > 1) adds nothing measurable at 512 datasets
(the residual integrated-likelihood bias is ≈ 0), and the SIR redraw (redraw='sir')
gives small, consistent, never-worse gains (largest at huge: σ_rfx EACE 0.034 vs
0.043) at ~1.7× the imhLaplace cost — both ship off by default.
"""

import math

import torch
from torch import Tensor
from torch import distributions as D
from torch.nn import functional as F

from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.families import POISSON_ETA_CLIP_MAX
from metabeta.utils.quadrature import ghProductGrid
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
    n_backtrack: int = 3,
    n_newton_extra: int = 15,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-group conditional modes and Hessians of p(rfx_j | θ_g, y_j).

    The Newton step backtracks (per (b, m, s) entry, up to n_backtrack halvings)
    whenever it would decrease the per-group objective ℓ_j(b) − ½ bᵀΣ⁻¹b. The target
    is strictly log-concave in b, so damped ascent converges globally; full steps
    can oscillate/diverge on extreme proposal samples (Poisson exp overshoot),
    which made the weight a function of the warm start — Δlog w between two inits
    reached 2e4–7e4 nats on samples near the pool max weight (2026-09-12
    newton_stability diagnostic), i.e. init-dependent absorbing states in IMH and
    poisoned SNIS weights.

    Returns (modes, chol_H, Sigma_inv, L_rfx, decrement):
        modes    (b, m, s, q)     — Newton solution b*_j
        chol_H   (b, m, s, q, q)  — Cholesky of H_j = ZᵀW(b*)Z + Σ⁻¹
        Sigma_inv (b, s, q, q), L_rfx (b, s, q, q) — reusable Σ_rfx factors
        decrement (b, m, s)       — final Newton decrement λ²/2: the unresolved
            objective error in nats. Large values mark (sample, group) pairs whose
            Laplace weight is numerically meaningless (huge-count Poisson datasets
            plateau the clipped objective and defeat even backtracked Newton).
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
    modes = (modes * mask_mq).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    def hessian(w: Tensor) -> Tensor:
        ZWZ = torch.einsum('bmns,bmnq,bmnr->bmsqr', w * mask_n, Z_m, Z_m)
        return ZWZ + Sigma_inv.unsqueeze(1)

    def objective(cand: Tensor) -> Tensor:
        """Per-group log target (up to θ_g-constants): ℓ_j(cand) − ½ candᵀΣ⁻¹cand."""
        eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, cand)
        ll = _llPerGroup(eta, y, sigma_eps, mask_n, likelihood_family)
        quad = torch.einsum('bmsq,bsqr,bmsr->bms', cand, Sigma_inv, cand)
        return ll - 0.5 * quad

    # Adaptive iteration count: the per-iteration Newton decrement λ²/2 comes for
    # free from (score, delta), so after the standard n_newton steps the loop keeps
    # going (up to n_newton_extra more) only while some entry is still unresolved —
    # hard samples (huge-count Poisson) get the budget they need, clean datasets pay
    # nothing. The loop breaks BEFORE stepping, so chol_H/decrement are always
    # evaluated at the returned modes.
    tol = 0.1  # nats — resolve well below the 1-nat pinning guard downstream
    max_iter = n_newton + n_newton_extra + 1
    for t in range(max_iter):
        eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, modes)
        score_res, w = _meanWeightScore(eta, y, sigma_eps, likelihood_family)
        score = torch.einsum('bmnq,bmns->bmsq', Z_m, score_res * mask_n)
        score = score - torch.einsum('bsqr,bmsr->bmsq', Sigma_inv, modes)
        chol_H = torch.linalg.cholesky(hessian(w) + 1e-6 * eye)
        delta = torch.cholesky_solve(score.unsqueeze(-1), chol_H).squeeze(-1)
        decrement = (0.5 * (score * delta).sum(-1)).nan_to_num(nan=torch.inf) * mask_m
        if t == max_iter - 1 or (t >= n_newton - 1 and float(decrement.max()) <= tol):
            break
        delta = (damping * delta).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # backtracking line search per (b, m, s): halve entries whose step decreases
        # the objective (NaN counts as a decrease)
        obj0 = _llPerGroup(eta, y, sigma_eps, mask_n, likelihood_family) - 0.5 * torch.einsum(
            'bmsq,bsqr,bmsr->bms', modes, Sigma_inv, modes
        )
        for _ in range(n_backtrack):
            worse = ~(objective(modes + delta) >= obj0 - 1e-6)  # (b, m, s)
            if not worse.any():
                break
            delta = torch.where(worse.unsqueeze(-1), 0.5 * delta, delta)

        modes = (modes + delta) * mask_mq
        modes = modes.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    return modes, chol_H, Sigma_inv, L_rfx, decrement


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
    guard_nats: float | None = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Laplace-approximated marginal log-likelihood Σ_j log p̂(y_j | θ_g).

    log p̂(y_j|θ_g) = ℓ_j(b*) + log N(b*; 0, Σ) + (q/2) log 2π − ½ log det H_j
                   = ℓ_j(b*) − ½ (log det Σ + b*ᵀ Σ⁻¹ b* + log det H_j)

    (the two (q/2)·log 2π terms cancel; padded rfx dims cancel between the two
    log-dets exactly as in logMarginalLikelihoodNormal). Exact for Normal.

    Samples whose summed Newton decrement exceeds guard_nats (unresolved mode-search
    error, in nats) get their ll pinned to −1e10: their weight is numerically
    meaningless, and left alone such samples become init-dependent absorbing states
    in IMH / poisoned SNIS weights (2026-09-12 newton_stability diagnostic; worst on
    huge-count Poisson datasets). Pass guard_nats=None to disable.

    Returns (ll (b, s), modes, chol_H) — modes/chol_H reusable for the
    conditional redraw so the weights and rfx draws share one target.
    """
    modes, chol_H, Sigma_inv, L_rfx, decrement = laplaceRfxModes(
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
    ll = (laplace_g * mask_m).sum(dim=1)  # (b, s)
    if guard_nats is not None:
        unresolved = decrement.sum(dim=1)  # (b, s); decrement is masked per group
        ll = torch.where(unresolved > guard_nats, torch.full_like(ll, -1e10), ll)
    return ll, modes, chol_H


def logMarginalLikelihoodAGQ(
    ffx: Tensor,  # (b, s, d)
    sigma_rfx: Tensor,  # (b, s, q) — all q dims must be active (slice before calling)
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
    k: int = 9,  # Gauss-Hermite nodes per rfx dim (K = k^q total)
    guard_nats: float | None = 1.0,
) -> Tensor:
    """Adaptive Gauss-Hermite (nAGQ=k) marginal log-likelihood Σ_j log p̂(y_j | θ_g).

    The Laplace (nAGQ=1) integrated likelihood carries the classic downward σ_rfx
    bias for binary/count data with small groups; AGQ shrinks it at k^q extra bare
    likelihood evaluations. Substituting α = b* + √2·L_H⁻ᵀ z around the Newton mode:

        log p̂(y_j|θ_g) = (q/2)·log 2 − ½·log det H_j
                        + logsumexp_k [log w_k + ‖z_k‖² + ℓ_j(α_k) + log N(α_k; 0, Σ)]

    Exact for Normal at any k (the transformed integrand is constant). Unlike the
    Laplace formula there is no padded-dim cancellation here — callers must slice
    all inputs to the active rfx dims (see LaplaceImportanceSampler). Returns (b, s).
    """
    b, s, q = sigma_rfx.shape
    modes, chol_H, Sigma_inv, L_rfx, decrement = laplaceRfxModes(
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
    mu_ffx = torch.einsum('bmnd,bsd->bmns', X, ffx)  # (b, m, n, s)

    diag_L = L_rfx.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
    log_det_Sigma = 2.0 * diag_L.log().sum(-1)  # (b, s)
    log_det_H = 2.0 * chol_H.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (b, m, s)
    log_norm = -0.5 * (q * math.log(2.0 * math.pi) + log_det_Sigma)  # (b, s)

    z_nodes, log_w = ghProductGrid([k] * q, sigma_rfx.dtype, sigma_rfx.device)  # (K, q), (K,)
    # offsets α_k − b* = √2·L_H⁻ᵀ z_k, one triangular solve for all nodes
    rhs = z_nodes.T.expand(*chol_H.shape[:3], q, z_nodes.shape[0])
    offsets = math.sqrt(2.0) * torch.linalg.solve_triangular(chol_H.mT, rhs, upper=True)

    # node loop keeps the (b, m, n, s) likelihood tensor as the peak allocation
    summands = []
    for kk in range(z_nodes.shape[0]):
        alpha = modes + offsets[..., kk]  # (b, m, s, q)
        eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, alpha)
        ll_g = _llPerGroup(eta, y, sigma_eps, mask_n, likelihood_family)  # (b, m, s)
        quad = torch.einsum('bmsq,bsqr,bmsr->bms', alpha, Sigma_inv, alpha)
        log_prior = log_norm[:, None, :] - 0.5 * quad
        correction = log_w[kk] + z_nodes[kk].square().sum()
        summands.append(correction + ll_g + log_prior)

    lse = torch.logsumexp(torch.stack(summands, dim=-1), dim=-1)  # (b, m, s)
    lml_g = lse + 0.5 * q * math.log(2.0) - 0.5 * log_det_H
    ll = (lml_g * mask_m).sum(dim=1)  # (b, s)
    if guard_nats is not None:
        # same non-convergence guard as logMarginalLikelihoodLaplace: an AGQ grid
        # centred on a garbage mode is garbage too
        unresolved = decrement.sum(dim=1)  # (b, s)
        ll = torch.where(unresolved > guard_nats, torch.full_like(ll, -1e10), ll)
    return ll


def sampleRfxSIR(
    ffx: Tensor,  # (b, s, d)
    sigma_eps: Tensor,  # (b, s)
    y: Tensor,  # (b, m, n, 1)
    X: Tensor,  # (b, m, n, d)
    Z: Tensor,  # (b, m, n, q)
    mask_n: Tensor,  # (b, m, n, 1)
    mask_m: Tensor,  # (b, m, 1)
    likelihood_family: int,
    modes: Tensor,  # (b, m, s, q) — Laplace conditional modes
    chol_H: Tensor,  # (b, m, s, q, q)
    L_rfx: Tensor,  # (b, s, q, q) — Cholesky of Σ_rfx
    Sigma_inv: Tensor,  # (b, s, q, q)
    n_redraw: int = 16,
    w_laplace: float = 0.9,
) -> Tensor:
    """Skew-correcting conditional redraw: per-group SIR targeting p(rfx_j | θ_g, y_j).

    The exact conditional is strictly log-concave (no fat tails) but skewed where the
    Gaussian N(b*, H⁻¹) is symmetric — all-0/all-1 Bernoulli groups, low-count Poisson.
    n_redraw candidates per (dataset, group, sample) are drawn from the defensive
    mixture w·N(b*, H⁻¹) + (1−w)·N(0, Σ_rfx); log-weights use the exact unnormalized
    conditional ℓ_j(α) + log N(α; 0, Σ). The bounded GLM likelihood plus the prior
    mixture component bound the weights, so no candidate set can degenerate. One
    candidate per (b, m, s) is kept via streaming Gumbel-max. Returns (b, m, s, q).
    """
    b, m, s, q = modes.shape
    Z_m = Z * mask_n
    mu_ffx = torch.einsum('bmnd,bsd->bmns', X, ffx)  # (b, m, n, s)

    diag_L = L_rfx.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
    log_det_Sigma = 2.0 * diag_L.log().sum(-1)  # (b, s)
    log_det_H = 2.0 * chol_H.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (b, m, s)
    c_norm = q * math.log(2.0 * math.pi)
    log_w_lap = math.log(w_laplace)
    log_w_pri = math.log(1.0 - w_laplace)

    def logPrior(alpha: Tensor) -> Tensor:
        """log N(alpha; 0, Σ_rfx) per group — also the prior mixture-component density."""
        quad = torch.einsum('bmsq,bsqr,bmsr->bms', alpha, Sigma_inv, alpha)
        return -0.5 * (c_norm + log_det_Sigma[:, None, :] + quad)

    best_key = modes.new_full((b, m, s), -torch.inf)
    best = torch.zeros_like(modes)
    for _ in range(n_redraw):
        # candidate from the defensive mixture
        eps = torch.randn_like(modes)
        cand_lap = modes + torch.linalg.solve_triangular(
            chol_H.mT, eps.unsqueeze(-1), upper=True
        ).squeeze(-1)
        cand_pri = torch.einsum('bsqr,bmsr->bmsq', L_rfx, torch.randn_like(modes))
        take_lap = torch.rand(b, m, s, 1, device=modes.device) < w_laplace
        alpha = torch.where(take_lap, cand_lap, cand_pri)

        # exact unnormalized conditional
        eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, alpha)
        ll_g = _llPerGroup(eta, y, sigma_eps, mask_n, likelihood_family)  # (b, m, s)
        log_target = ll_g + logPrior(alpha)

        # mixture proposal density
        diff = alpha - modes
        v = torch.einsum('bmsqr,bmsq->bmsr', chol_H, diff)  # L_Hᵀ diff
        log_lap = -0.5 * (c_norm - log_det_H + v.square().sum(-1))
        log_mix = torch.logaddexp(log_w_lap + log_lap, log_w_pri + logPrior(alpha))

        log_weight = (log_target - log_mix).nan_to_num(nan=-torch.inf, posinf=-torch.inf)
        gumbel = -torch.rand(b, m, s, device=modes.device).log().neg().log()
        key = log_weight + gumbel
        replace = key > best_key
        best_key = torch.where(replace, key, best_key)
        best = torch.where(replace.unsqueeze(-1), alpha, best)

    return best * mask_m.unsqueeze(-1)


class LaplaceImportanceSampler(ImportanceSampler):
    """SNIS with Laplace marginal weights and Laplace conditional rfx redraw.

    The non-Normal analog of ImportanceSampler(marginal=True, rb_redraw=True):
    weights use the Laplace-approximated integrated likelihood, and rfx are
    redrawn from the Gaussian Laplace conditional N(b*, H⁻¹) at each global
    sample. With attach_only=True the weights are discarded (uniform) and only
    the rfx replacement is kept — zero weight bias at the cost of leaving global
    parameters uncorrected.

    Optional upgrades (both compose; see LAPLACE_UPGRADES.md in experiments/posthoc):
    - nagq > 1: datasets whose active rfx dimension is ≤ agq_max_q get adaptive
      Gauss-Hermite marginal weights (nagq nodes/dim, capped at 7 for q=2) instead
      of Laplace — targets the downward σ_rfx weight bias. Other datasets keep
      Laplace weights; the conditional redraw is unaffected either way.
    - redraw='sir': the conditional redraw becomes a per-group SIR pass targeting
      the exact conditional (n_redraw candidates from a defensive Laplace+prior
      mixture) — captures the skew that the symmetric N(b*, H⁻¹) draw misses.
    """

    def __init__(
        self,
        data: dict[str, Tensor],
        attach_only: bool = False,
        n_newton: int = 5,
        nagq: int = 1,
        agq_max_q: int = 2,
        redraw: str = 'laplace',
        n_redraw: int = 16,
        **kwargs,
    ) -> None:
        if kwargs.get('marginal') or kwargs.get('full'):
            raise ValueError('LaplaceImportanceSampler defines its own marginal weights')
        if redraw not in ('laplace', 'sir'):
            raise ValueError(f"redraw must be 'laplace' or 'sir', got {redraw!r}")
        rb_redraw = kwargs.pop('rb_redraw', True)
        super().__init__(data, **kwargs)
        self.rb_redraw = rb_redraw  # bypass parent's marginal-only validation
        self.attach_only = attach_only
        self.n_newton = n_newton
        self.nagq = nagq
        self.agq_max_q = agq_max_q
        self.redraw = redraw
        self.n_redraw = n_redraw
        self._modes: Tensor | None = None
        self._chol_H: Tensor | None = None
        self._ffx: Tensor | None = None
        self._sigma_eps: Tensor | None = None

    def _agqRows(self, qa: int) -> Tensor:
        """Datasets whose active rfx dims are exactly the leading qa dims. (b,) bool."""
        mask_q = self.mask_q.squeeze(-2)  # (b, q)
        q = mask_q.shape[-1]
        leading = mask_q[:, :qa].all(dim=-1) if qa > 0 else torch.ones_like(mask_q[:, 0])
        trailing = mask_q[:, qa:].any(dim=-1) if qa < q else torch.zeros_like(leading)
        return leading.bool() & ~trailing.bool()

    def unnormalizedPosterior(self, proposal: Proposal) -> tuple[Tensor, Tensor]:
        lp, ffx, sigma_eps = self._logPriorGlobals(proposal)
        L_corr = self._getLCorr(proposal)
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
            L_corr=L_corr,
            init=proposal.rfx,
            n_newton=self.n_newton,
        )
        # AGQ weight upgrade for low-dimensional rfx (grouped by active q so each
        # group integrates only its own dims; padded dims are sliced away)
        if self.nagq > 1:
            for qa in range(1, min(self.agq_max_q, proposal.q) + 1):
                rows = self._agqRows(qa)
                if not rows.any():
                    continue
                k = self.nagq if qa == 1 else min(self.nagq, 7)
                ll_agq = logMarginalLikelihoodAGQ(
                    ffx[rows],
                    proposal.sigma_rfx[rows][..., :qa],
                    sigma_eps[rows],
                    self.y[rows],
                    self.X[rows],
                    self.Z[rows][..., :qa],
                    self.mask_n[rows],
                    self.mask_m[rows],
                    self.likelihood_family,
                    L_corr=None if L_corr is None else L_corr[rows][..., :qa, :qa],
                    init=proposal.rfx[rows][..., :qa],
                    n_newton=self.n_newton,
                    k=k,
                )
                ll[rows] = ll_agq
        self._modes, self._chol_H = modes, chol_H
        self._ffx, self._sigma_eps = ffx, sigma_eps
        return ll, lp

    def _redrawRfx(self, proposal: Proposal) -> None:
        if self.redraw == 'sir':
            L_corr = self._getLCorr(proposal)
            L_rfx = _sigmaChol(proposal.sigma_rfx, L_corr)
            eye = torch.eye(L_rfx.shape[-1], dtype=L_rfx.dtype, device=L_rfx.device)
            Sigma_inv = torch.cholesky_solve(eye.expand_as(L_rfx), L_rfx)
            rfx = sampleRfxSIR(
                self._ffx,
                self._sigma_eps,
                self.y,
                self.X,
                self.Z,
                self.mask_n,
                self.mask_m,
                self.likelihood_family,
                self._modes,
                self._chol_H,
                L_rfx,
                Sigma_inv,
                n_redraw=self.n_redraw,
            )
        else:
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
