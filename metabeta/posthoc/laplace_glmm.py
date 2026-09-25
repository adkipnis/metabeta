"""
posthoc/laplace_glmm.py — Laplace-approximated Rao-Blackwellisation for non-Normal GLMMs.

Non-Normal likelihoods have no conjugate rfx conditional and no closed-form marginal
likelihood, so the exact Rao-Blackwellised SNIS of posthoc/importance.py (marginal=True)
does not apply. This module provides the Laplace analog, fully vectorized over
(datasets b, groups m, posterior samples s):

1. `laplaceRfxModes` — damped-Newton per-group conditional modes b*_j and Hessians
   H_j = ZᵀWZ + Σ⁻¹ of p(rfx_j | θ_g, y_j), started from the proposal's rfx: the flow's
   draws, or the analytical rfx estimate when the local flow was skipped at inference
   (Approximator.estimate(local=False)). The log-concave target converges from either;
   the start affects only speed and, at the finite budget, the guard-pinned tail.
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

Findings (2026-09, 512 test datasets)
-------------------------------------
Most of the σ_rfx shift above was Newton-instability contamination of the weights, not
Laplace bias: full-step Newton oscillated on extreme proposals (huge-count Poisson),
making log p̂(y|θ_g) depend on the warm start by 1e4+ nats on top-weight samples. With the
robustified mode search below, Poisson-large isLaplace σ_rfx ECE went −0.046 → −0.031
(raw −0.027) and Bernoulli-huge max PSIS k 15.4 → 2.6.
"""

import math

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


def _llScoreWeight(
    eta: Tensor,  # (b, m, n, s)
    y: Tensor,  # (b, m, n, 1)
    sigma_eps: Tensor,  # (b, s)
    mask_n: Tensor,  # (b, m, n, 1)
    likelihood_family: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused _llPerGroup + _meanWeightScore: (ll (b, m, s), score_res, w) from one pass over η.

    The Newton step needs the score and weights at the current modes and the backtracking
    line search needs the objective there; computing both from the same μ saves a full
    (b, m, n, s) likelihood pass per iteration.
    """
    if likelihood_family == 0:
        phi_inv = (1.0 / sigma_eps.pow(2).clamp(min=1e-12))[:, None, None, :]
        scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
        ll = D.Normal(loc=eta, scale=scale).log_prob(y)
        score_res, w = (y - eta) * phi_inv, phi_inv.expand_as(eta)
    elif likelihood_family == 1:
        mu = torch.sigmoid(eta)
        ll = y * eta - F.softplus(eta)
        score_res, w = y - mu, (mu * (1.0 - mu)).clamp(min=1e-6)
    elif likelihood_family == 2:
        eta_c = eta.clamp(max=POISSON_ETA_CLIP_MAX)
        mu = torch.exp(eta_c)
        ll = y * eta_c - mu - torch.lgamma(y + 1.0)
        score_res, w = y - mu, mu.clamp(min=1e-6)
    else:
        raise NotImplementedError(f'likelihood_family={likelihood_family}')
    return (ll * mask_n).sum(dim=2), score_res, w


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
    n_newton: int = 3,  # standard damped-Newton steps; stragglers go to the compacted extra
    # phase, so 3 matches 5 to <=0.002 on every recovery/calibration metric (512-dataset check,
    # bernoulli/poisson x 4 sizes, 2026-09-23) at ~15-35% less CPU — the mode search is ~95%
    # of the imhLaplace refinement. Purely post-hoc: does NOT touch the analytical context stats.
    damping: float = 1.0,
    n_backtrack: int = 3,  # do not lower: bt<=1 diverges on deep/extreme entries (guard-pins)
    n_newton_extra: int = 15,
    tol: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-group conditional modes and Hessians of p(rfx_j | θ_g, y_j).

    The Newton step backtracks (per (b, m, s) entry, up to n_backtrack halvings) whenever
    it would decrease the per-group objective ℓ_j(b) − ½ bᵀΣ⁻¹b; the target is strictly
    log-concave in b, so damped ascent converges globally, whereas full steps oscillate on
    extreme Poisson proposals and made the weight warm-start-dependent.

    Returns (modes, chol_H, Sigma_inv, L_rfx, decrement):
        modes    (b, m, s, q)     — Newton solution b*_j
        chol_H   (b, m, s, q, q)  — Cholesky of H_j = ZᵀW(b*)Z + Σ⁻¹
        Sigma_inv (b, s, q, q), L_rfx (b, s, q, q) — reusable Σ_rfx factors
        decrement (b, m, s)       — final Newton decrement λ²/2, the unresolved objective
            error in nats; large values mark entries whose Laplace weight is meaningless.
    ``tol`` is the per-entry resolution threshold on that decrement (nats).
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

    # ZᵀWZ as one contraction over n of the s-free outer products (b, m, n, q, q) with the
    # weights (b, m, n, s): the three-operand einsum used to materialise a (b, m, n, s, q)
    # intermediate, q times the size of the largest tensor in the pass.
    ZZ = torch.einsum('bmnq,bmnr->bmnqr', Z_m, Z_m)

    def hessian(w: Tensor) -> Tensor:
        ZWZ = torch.einsum('bmnqr,bmns->bmsqr', ZZ, w * mask_n)
        return ZWZ + Sigma_inv.unsqueeze(1)

    def objective(cand: Tensor) -> Tensor:
        """Per-group log target (up to θ_g-constants): ℓ_j(cand) − ½ candᵀΣ⁻¹cand."""
        eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, cand)
        ll = _llPerGroup(eta, y, sigma_eps, mask_n, likelihood_family)
        quad = torch.einsum('bmsq,bsqr,bmsr->bms', cand, Sigma_inv, cand)
        return ll - 0.5 * quad

    # Standard phase: n_newton damped steps on every (b, m, s) entry, then one more
    # score/Hessian evaluation at the modes we return (chol_H/decrement always belong to
    # them). The Newton decrement λ²/2 comes for free from (score, delta).
    # tol [nats]: an entry counts as resolved once its Newton decrement λ²/2 (the
    # remaining objective error) is below it — well under the 1-nat pinning guard downstream
    for t in range(n_newton + 1):
        eta = mu_ffx + torch.einsum('bmnq,bmsq->bmns', Z_m, modes)
        ll0, score_res, w = _llScoreWeight(eta, y, sigma_eps, mask_n, likelihood_family)
        score = torch.einsum('bmnq,bmns->bmsq', Z_m, score_res * mask_n)
        score = score - torch.einsum('bsqr,bmsr->bmsq', Sigma_inv, modes)
        chol_H = torch.linalg.cholesky(hessian(w) + 1e-6 * eye)
        delta = torch.cholesky_solve(score.unsqueeze(-1), chol_H).squeeze(-1)
        decrement = (0.5 * (score * delta).sum(-1)).nan_to_num(nan=torch.inf) * mask_m
        if t == n_newton or (t >= n_newton - 1 and float(decrement.max()) <= tol):
            break
        delta = (damping * delta).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # backtracking line search per (b, m, s): halve entries whose step decreases
        # the objective (NaN counts as a decrease)
        obj0 = ll0 - 0.5 * torch.einsum('bmsq,bsqr,bmsr->bms', modes, Sigma_inv, modes)
        for _ in range(n_backtrack):
            worse = ~(objective(modes + delta) >= obj0 - 1e-6)  # (b, m, s)
            if not worse.any():
                break
            delta = torch.where(worse.unsqueeze(-1), 0.5 * delta, delta)

        modes = (modes + delta) * mask_mq
        modes = modes.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    # Extra phase: only the entries still above tol keep iterating (up to n_newton_extra
    # more steps), compacted to a (K, n, ·) layout. After the standard phase these are
    # 0.1–2% of the entries (huge-count Poisson proposals), so a global rule that kept
    # every entry stepping until the last one resolved cost ~3x the whole pass.
    unresolved = decrement > tol
    if n_newton_extra > 0 and bool(unresolved.any()):
        modes, chol_H, decrement = _refineUnresolved(
            unresolved,
            modes,
            chol_H,
            decrement,
            mu_ffx,
            Z_m,
            y,
            mask_n,
            sigma_eps,
            Sigma_inv,
            likelihood_family,
            n_newton_extra,
            damping,
            n_backtrack,
            tol,
        )

    return modes, chol_H, Sigma_inv, L_rfx, decrement


def _refineUnresolved(
    unresolved: Tensor,  # (b, m, s) bool
    modes: Tensor,  # (b, m, s, q)
    chol_H: Tensor,  # (b, m, s, q, q)
    decrement: Tensor,  # (b, m, s)
    mu_ffx: Tensor,  # (b, m, n, s)
    Z_m: Tensor,  # (b, m, n, q), padded observations already zeroed
    y: Tensor,  # (b, m, n, 1)
    mask_n: Tensor,  # (b, m, n, 1)
    sigma_eps: Tensor,  # (b, s)
    Sigma_inv: Tensor,  # (b, s, q, q)
    likelihood_family: int,
    n_extra: int,
    damping: float,
    n_backtrack: int,
    tol: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Continue the damped Newton search of laplaceRfxModes on the unresolved entries only.

    The K unresolved (b, m, s) entries are gathered into (K, n, ·) tensors — each entry's
    group data, fixed-effect offset and Σ⁻¹ — and iterated with the same step, backtracking
    and clamping as the standard phase. Entries that resolve are frozen (zero step); the loop
    ends when all are resolved or the extra budget is spent. Results are scattered back, so
    (modes, chol_H, decrement) stay consistent at the returned modes.
    """
    bi, mi, si = unresolved.nonzero(as_tuple=True)  # (K,)
    q = modes.shape[-1]
    Zk = Z_m[bi, mi]  # (K, n, q)
    yk = y[bi, mi, :, 0]  # (K, n)
    nk = mask_n[bi, mi, :, 0]  # (K, n)
    muk = mu_ffx[bi, mi, :, si]  # (K, n)
    sek = sigma_eps[bi, si]  # (K,)
    Sik = Sigma_inv[bi, si]  # (K, q, q)
    modk = modes[bi, mi, si]  # (K, q)
    eye = torch.eye(q, dtype=modk.dtype, device=modk.device)

    def fused(eta_k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(ll (K,), score_res (K, n), w (K, n)) at η via the family helper in its
        (b=K, m=1, s=1) layout, so the per-entry σ_eps broadcasts like a (b, s) tensor."""
        ll, r, w = _llScoreWeight(
            eta_k[:, None, :, None],
            yk[:, None, :, None],
            sek[:, None],
            nk[:, None, :, None],
            likelihood_family,
        )
        return ll[:, 0, 0], r[:, 0, :, 0], w[:, 0, :, 0]

    def objective(cand: Tensor) -> Tensor:
        eta_k = muk + torch.einsum('knq,kq->kn', Zk, cand)
        return fused(eta_k)[0] - 0.5 * torch.einsum('kq,kqr,kr->k', cand, Sik, cand)

    for t in range(n_extra + 1):
        eta_k = muk + torch.einsum('knq,kq->kn', Zk, modk)
        ll0, score_res, w = fused(eta_k)
        score = torch.einsum('knq,kn->kq', Zk, score_res * nk) - torch.einsum(
            'kqr,kr->kq', Sik, modk
        )
        H = torch.einsum('kn,knq,knr->kqr', w * nk, Zk, Zk) + Sik
        chol = torch.linalg.cholesky(H + 1e-6 * eye)
        delta = torch.cholesky_solve(score.unsqueeze(-1), chol).squeeze(-1)
        dec = (0.5 * (score * delta).sum(-1)).nan_to_num(nan=torch.inf)
        active = dec > tol
        if t == n_extra or not bool(active.any()):
            break
        delta = (damping * delta).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        delta = torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))
        obj0 = ll0 - 0.5 * torch.einsum('kq,kqr,kr->k', modk, Sik, modk)
        for _ in range(n_backtrack):
            worse = ~(objective(modk + delta) >= obj0 - 1e-6)
            if not bool(worse.any()):
                break
            delta = torch.where(worse.unsqueeze(-1), 0.5 * delta, delta)
        modk = (modk + delta).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    modes, chol_H, decrement = modes.clone(), chol_H.clone(), decrement.clone()
    modes[bi, mi, si] = modk
    chol_H[bi, mi, si] = chol
    decrement[bi, mi, si] = dec
    return modes, chol_H, decrement


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
    n_newton: int = 3,  # see laplaceRfxModes: 3 == 5 to <=0.002 on all metrics, ~15-35% cheaper
    guard_nats: float | None = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Laplace-approximated marginal log-likelihood Σ_j log p̂(y_j | θ_g).

    log p̂(y_j|θ_g) = ℓ_j(b*) + log N(b*; 0, Σ) + (q/2) log 2π − ½ log det H_j
                   = ℓ_j(b*) − ½ (log det Σ + b*ᵀ Σ⁻¹ b* + log det H_j)

    (the two (q/2)·log 2π terms cancel; padded rfx dims cancel between the two
    log-dets exactly as in logMarginalLikelihoodNormal). Exact for Normal.

    Samples whose summed Newton decrement exceeds guard_nats get their ll pinned to −1e10:
    an unresolved mode search makes the weight meaningless, and such samples otherwise
    become init-dependent absorbing states in IMH. guard_nats=None disables the guard.

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


class _GroupIntegrand:
    """Per-group integrand of p(y_j | θ_g) = ∫ p(y_j | θ_g, b) N(b; 0, Σ) db around its Laplace
    Gaussian N(b*_j, H_j⁻¹), shared by the IS² estimate and AGQ.

    Owns the whitening: candidates are b = b* + U⁻ᵀ z (H = U Uᵀ) or prior draws L z, and
    `logWeight` returns log p(y_j, b | θ_g) − log r(b) for the proposal r = (1 − α) Laplace +
    α prior. The two Gaussians' (q/2)·log 2π terms cancel, and padded rfx dims cancel between
    their log-dets as in logMarginalLikelihoodLaplace.
    """

    def __init__(
        self,
        ffx: Tensor,  # (b, s, d)
        sigma_rfx: Tensor,  # (b, s, q)
        sigma_eps: Tensor,  # (b, s)
        y: Tensor,  # (b, m, n, 1)
        X: Tensor,  # (b, m, n, d)
        Z: Tensor,  # (b, m, n, q)
        mask_n: Tensor,  # (b, m, n, 1)
        mask_m: Tensor,  # (b, m, 1)
        likelihood_family: int,
        L_corr: Tensor | None,
        init: Tensor | None,
        n_newton: int,
        defensive: float,
    ) -> None:
        self.modes, self.chol_H, _, L_rfx, _ = laplaceRfxModes(
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
        self.y, self.sigma_eps, self.mask_n = y, sigma_eps, mask_n
        self.likelihood_family = likelihood_family
        self.defensive = defensive
        self.Z_m = Z * mask_n
        self.mu_ffx = torch.einsum('bmnd,bsd->bmns', X, ffx)  # (b, m, n, s)
        self.log_det_U = self.chol_H.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (b, m, s)
        log_det_L = L_rfx.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).log().sum(-1)
        self.log_det_L = log_det_L[:, None]  # (b, 1, s)
        self.L_rfx = L_rfx.unsqueeze(1)  # (b, 1, s, q, q)

    def laplaceDraw(self, z: Tensor) -> Tensor:
        """b* + U⁻ᵀ z for standard-normal (or quadrature-node) z (b, m, s, q)."""
        shift = torch.linalg.solve_triangular(self.chol_H.mT, z.unsqueeze(-1), upper=True)
        return self.modes + shift.squeeze(-1)

    def priorDraw(self, z: Tensor) -> Tensor:
        return (self.L_rfx @ z.unsqueeze(-1)).squeeze(-1)

    def logWeight(self, cand: Tensor) -> Tensor:
        """log p(y_j, cand | θ_g) − log r(cand), (b, m, s)."""
        white_lap = (self.chol_H.mT @ (cand - self.modes).unsqueeze(-1)).squeeze(-1)
        white_pri = torch.linalg.solve_triangular(self.L_rfx, cand.unsqueeze(-1), upper=False)
        log_lap = -0.5 * white_lap.square().sum(-1) + self.log_det_U
        log_pri = -0.5 * white_pri.squeeze(-1).square().sum(-1) - self.log_det_L
        log_r = log_lap
        if self.defensive > 0:
            log_r = torch.logaddexp(
                math.log1p(-self.defensive) + log_lap, math.log(self.defensive) + log_pri
            )
        eta = self.mu_ffx + torch.einsum('bmnq,bmsq->bmns', self.Z_m, cand)
        ll = _llPerGroup(eta, self.y, self.sigma_eps, self.mask_n, self.likelihood_family)
        return ll + log_pri - log_r


def logMarginalLikelihoodIS2(
    ffx: Tensor,  # (b, s, d)
    sigma_rfx: Tensor,  # (b, s, q)
    sigma_eps: Tensor,  # (b, s)
    y: Tensor,  # (b, m, n, 1)
    X: Tensor,  # (b, m, n, d)
    Z: Tensor,  # (b, m, n, q)
    mask_n: Tensor,  # (b, m, n, 1)
    mask_m: Tensor,  # (b, m, 1)
    likelihood_family: int,
    n_inner: int,
    L_corr: Tensor | None = None,
    init: Tensor | None = None,
    n_newton: int = 3,
    defensive: float = 0.01,  # is2_tuning: 0.01 ≈ 0 in sd(log p̂), 0.1 inflates it ~1.8x
) -> tuple[Tensor, Tensor]:
    """Unbiased estimate Σ_j log p̂(y_j | θ_g) by per-group importance sampling (IS²).

    Importance sampling squared (Tran, Scharth, Pitt & Kohn, arXiv:1309.3339): per group,
    p̂_j = (1/K) Σ_k p(y_j | θ_g, b_k) N(b_k; 0, Σ) / r_j(b_k) with b_k ~ r_j, so E[p̂_j] =
    p(y_j | θ_g) and, the groups being independent given θ_g, E[∏_j p̂_j] = p(y | θ_g).
    Plugging exp(ll) into the IS weights therefore keeps the evidence unbiased, and the IMH
    becomes pseudo-marginal (Andrieu & Roberts 2009), i.e. it targets the exact posterior.

    r_j is the defensive mixture (1 − α) N(b*_j, H_j⁻¹) + α N(0, Σ) (Hesterberg 1995): the
    Laplace Gaussian is lighter-tailed than the prior (H ⪰ Σ⁻¹), which leaves the inner
    weights with infinite variance where the likelihood is flat (all-success Bernoulli
    groups); the prior component bounds them by max_b p(y_j | θ_g, b) / α.

    Returns (ll (b, s), rfx (b, m, s, q)); rfx holds one inner draw per group chosen with
    probability ∝ its weight (single-item weighted reservoir sampling over k), an exact
    draw from p(rfx_j | θ_g, y_j) under the pseudo-marginal extended target.
    """
    f = _GroupIntegrand(
        ffx,
        sigma_rfx,
        sigma_eps,
        y,
        X,
        Z,
        mask_n,
        mask_m,
        likelihood_family,
        L_corr,
        init,
        n_newton,
        defensive,
    )
    log_sum = torch.full_like(f.log_det_U, -torch.inf)  # (b, m, s) running logsumexp
    rfx = torch.zeros_like(f.modes)  # (b, m, s, q)
    for _ in range(n_inner):
        z = torch.randn_like(f.modes)
        from_prior = torch.rand_like(f.log_det_U) < defensive  # (b, m, s)
        cand = torch.where(from_prior.unsqueeze(-1), f.priorDraw(z), f.laplaceDraw(z))
        log_w = f.logWeight(cand)
        log_sum_new = torch.logaddexp(log_sum, log_w)
        take = torch.rand_like(log_w) < torch.exp(log_w - log_sum_new)  # NaN (both −inf) → keep
        rfx = torch.where(take.unsqueeze(-1), cand, rfx)
        log_sum = log_sum_new

    ll = ((log_sum - math.log(n_inner)) * mask_m).sum(dim=1)  # (b, s)
    return ll, rfx * mask_m.unsqueeze(-1)


def logMarginalLikelihoodAGQ(
    ffx: Tensor,  # (b, s, d)
    sigma_rfx: Tensor,  # (b, s, q)
    sigma_eps: Tensor,  # (b, s)
    y: Tensor,  # (b, m, n, 1)
    X: Tensor,  # (b, m, n, d)
    Z: Tensor,  # (b, m, n, q)
    mask_n: Tensor,  # (b, m, n, 1)
    mask_m: Tensor,  # (b, m, 1)
    likelihood_family: int,
    n_nodes: int,
    L_corr: Tensor | None = None,
    init: Tensor | None = None,
    n_newton: int = 3,
) -> Tensor:
    """Σ_j log p(y_j | θ_g) by adaptive Gauss-Hermite quadrature (AGQ). Returns (b, s).

    Tensor-product probabilists' Gauss-Hermite nodes z_i, mapped through the Laplace Gaussian
    b_i = b* + U⁻ᵀ z_i (Liu & Pierce 1994; lme4's nAGQ): p(y_j | θ_g) ≈ Σ_i ν_i p(y_j, b_i | θ_g)
    / N(b_i; b*, H⁻¹), i.e. the IS² sum with deterministic nodes. n_nodes = 1 is the Laplace
    approximation; the error decays geometrically in n_nodes, which makes AGQ the evidence
    reference for GLMMs. Rfx dims unused by every dataset get a single node (exact there),
    so the cost is n_nodes^q_active likelihood passes.
    """
    from numpy.polynomial.hermite_e import hermegauss

    f = _GroupIntegrand(
        ffx,
        sigma_rfx,
        sigma_eps,
        y,
        X,
        Z,
        mask_n,
        mask_m,
        likelihood_family,
        L_corr,
        init,
        n_newton,
        defensive=0.0,
    )
    z_1d, w_1d = hermegauss(n_nodes)
    z_1d = torch.as_tensor(z_1d, dtype=ffx.dtype)
    log_w_1d = torch.as_tensor(w_1d / w_1d.sum(), dtype=ffx.dtype).log()
    active = (Z != 0).flatten(0, 2).any(0)  # (q,)
    axes = [(z_1d, log_w_1d) if a else (z_1d.new_zeros(1), log_w_1d.new_zeros(1)) for a in active]
    nodes = torch.cartesian_prod(*[z for z, _ in axes]).view(-1, len(axes))  # (P, q)
    log_nu = torch.cartesian_prod(*[w for _, w in axes]).view(-1, len(axes)).sum(-1)  # (P,)

    log_sum = torch.full_like(f.log_det_U, -torch.inf)  # (b, m, s)
    for z, lw in zip(nodes, log_nu):
        cand = f.laplaceDraw(z.expand_as(f.modes))
        log_sum = torch.logaddexp(log_sum, lw + f.logWeight(cand))
    return (log_sum * mask_m).sum(dim=1)


class LaplaceImportanceSampler(ImportanceSampler):
    """SNIS with Laplace marginal weights and Laplace conditional rfx redraw.

    The non-Normal analog of ImportanceSampler(marginal=True, rb_redraw=True):
    weights use the Laplace-approximated integrated likelihood, and rfx are
    redrawn from the Gaussian Laplace conditional N(b*, H⁻¹) at each global
    sample. With attach_only=True the weights are discarded (uniform) and only
    the rfx replacement is kept — zero weight bias at the cost of leaving global
    parameters uncorrected.

    n_inner=K > 0 swaps the Laplace marginal for its unbiased IS² estimate
    (logMarginalLikelihoodIS2): the weights then target the exact marginal posterior,
    log_evidence is unbiased on the evidence scale, and the rfx are the estimate's
    weight-selected inner draws instead of Laplace-Gaussian redraws.
    """

    def __init__(
        self,
        data: dict[str, Tensor],
        attach_only: bool = False,
        n_newton: int = 3,  # see laplaceRfxModes: 3 == 5 to <=0.002 on all metrics, ~15-35% cheaper
        n_inner: int = 0,  # IS² draws per group; 0 keeps the Laplace marginal
        **kwargs,
    ) -> None:
        if kwargs.get('marginal') or kwargs.get('full'):
            raise ValueError('LaplaceImportanceSampler defines its own marginal weights')
        rb_redraw = kwargs.pop('rb_redraw', True)
        super().__init__(data, **kwargs)
        self.rb_redraw = rb_redraw  # bypass parent's marginal-only validation
        self.attach_only = attach_only
        self.n_newton = n_newton
        self.n_inner = n_inner
        self._modes: Tensor | None = None
        self._chol_H: Tensor | None = None
        self._rfx: Tensor | None = None  # IS² inner draws, (b, m, s, q)

    def unnormalizedPosterior(self, proposal: Proposal) -> tuple[Tensor, Tensor]:
        lp, ffx, sigma_eps = self._logPriorGlobals(proposal)
        if self.n_inner > 0:
            ll, self._rfx = logMarginalLikelihoodIS2(
                ffx,
                proposal.sigma_rfx,
                sigma_eps,
                self.y,
                self.X,
                self.Z,
                self.mask_n,
                self.mask_m,
                self.likelihood_family,
                self.n_inner,
                L_corr=self._getLCorr(proposal),
                init=proposal.rfx,
                n_newton=self.n_newton,
            )
            return ll, lp
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
        if self.n_inner > 0:
            rfx = self._rfx
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
