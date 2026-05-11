"""REML/profile-MAP variance-scale refinements for Gaussian GLMM summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from metabeta.analytical.map import _fixedCorrFromStats, _logMarginalTarget, _replacePsiDiag

__all__ = [
    'NormalRemlGateConfig',
    'RemlRefineMeta',
    'gateNormalRemlVsMap',
    'refineNormalRemlSrfx',
]


@dataclass(frozen=True)
class RemlRefineMeta:
    """Row-level diagnostics for REML/profile-MAP variance refinement."""

    valid: torch.Tensor
    clamped: torch.Tensor


@dataclass(frozen=True)
class NormalRemlGateConfig:
    """Conservative row gate for using REML instead of current MAP output."""

    min_q: int = 2
    max_n_total: int = 1999


def _emptyMeta(stats: dict[str, torch.Tensor]) -> RemlRefineMeta:
    B = stats['sigma_rfx_est'].shape[0]
    device = stats['sigma_rfx_est'].device
    return RemlRefineMeta(
        valid=torch.zeros(B, dtype=torch.bool, device=device),
        clamped=torch.zeros(B, dtype=torch.bool, device=device),
    )


def refineNormalRemlSrfx(
    center: dict[str, torch.Tensor],
    fallback: dict[str, torch.Tensor],
    Xm: torch.Tensor,
    ym: torch.Tensor,
    Zm: torch.Tensor,
    mask_n: torch.Tensor,
    mask_m: torch.Tensor,
    nu_ffx: torch.Tensor,
    tau_ffx: torch.Tensor,
    family_ffx: torch.Tensor,
    tau_rfx: torch.Tensor,
    family_sigma_rfx: torch.Tensor,
    tau_eps: torch.Tensor,
    family_sigma_eps: torch.Tensor,
    eta_rfx: torch.Tensor | None = None,
    mask_d: torch.Tensor | None = None,
    mask_q: torch.Tensor | None = None,
    optimize_sigma_eps: bool = False,
    n_steps: int = 20,
    lr: float = 0.03,
) -> tuple[dict[str, torch.Tensor], RemlRefineMeta]:
    """Refine variance scales with beta/correlation fixed and row-level fallback.

    `center` supplies the MoM/EM initialization. `fallback` supplies the stats kept
    for invalid rows, normally the current production MAP result. FFX, BLUP, and
    by default sEps are inherited from `fallback`.
    """
    q = Zm.shape[-1]
    if q == 0 or n_steps <= 0:
        return fallback, _emptyMeta(fallback)

    corr = _fixedCorrFromStats(center, eta_rfx, mask_q, q).detach()
    beta = center['beta_est'].detach()
    log_sigma_rfx = (
        center['sigma_rfx_est'].detach().clamp(min=1e-4, max=20.0).log().clone()
    ).requires_grad_(True)
    log_sigma_eps = (
        center['sigma_eps_est'].squeeze(-1).detach().clamp(min=1e-4, max=20.0).log().clone()
    )
    params: list[torch.Tensor] = [log_sigma_rfx]
    if optimize_sigma_eps:
        log_sigma_eps = log_sigma_eps.clone().requires_grad_(True)
        params.append(log_sigma_eps)

    optimizer = torch.optim.Adam(params, lr=lr)
    valid_rows = torch.ones(beta.shape[0], dtype=torch.bool, device=beta.device)
    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            target = _logMarginalTarget(
                beta.unsqueeze(1),
                log_sigma_rfx.unsqueeze(1),
                log_sigma_eps.unsqueeze(1),
                corr,
                Xm[..., : beta.shape[-1]],
                ym,
                Zm,
                mask_n.float(),
                mask_m.float(),
                nu_ffx,
                tau_ffx,
                family_ffx,
                tau_rfx,
                family_sigma_rfx,
                tau_eps,
                family_sigma_eps,
                mask_d,
                mask_q,
            ).squeeze(1)
            finite_target = torch.isfinite(target)
            active_rows = valid_rows & finite_target
            valid_rows = valid_rows & finite_target
            if not bool(active_rows.any()):
                break
            loss = -target[active_rows].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()
            with torch.no_grad():
                log_sigma_rfx.clamp_(math.log(1e-4), math.log(20.0))
                if optimize_sigma_eps:
                    log_sigma_eps.clamp_(math.log(1e-4), math.log(20.0))

    sigma_rfx = log_sigma_rfx.detach().exp()
    sigma_eps = log_sigma_eps.detach().exp()
    valid = valid_rows & torch.isfinite(sigma_rfx).all(dim=-1)
    valid = valid & (sigma_rfx >= 1e-4).all(dim=-1) & (sigma_rfx <= 20.0).all(dim=-1)
    if optimize_sigma_eps:
        valid = valid & torch.isfinite(sigma_eps) & (sigma_eps >= 1e-4) & (sigma_eps <= 20.0)

    if mask_q is not None:
        active_q = mask_q[..., :q].bool()
        sigma_rfx = torch.where(active_q, sigma_rfx, fallback['sigma_rfx_est'][..., :q])
    else:
        active_q = torch.ones_like(sigma_rfx, dtype=torch.bool)
    eps = sigma_rfx.new_tensor(1e-6)
    clamped = ((sigma_rfx <= 1e-4 + eps) | (sigma_rfx >= 20.0 - eps)) & active_q
    clamped = clamped.any(dim=-1)
    sigma_rfx = torch.where(valid[:, None], sigma_rfx, fallback['sigma_rfx_est'][..., :q])

    out = dict(fallback)
    out['sigma_rfx_est'] = sigma_rfx
    if optimize_sigma_eps:
        sigma_eps = torch.where(valid, sigma_eps, fallback['sigma_eps_est'].squeeze(-1))
        out['sigma_eps_est'] = sigma_eps.unsqueeze(-1)
    if 'Psi' in fallback:
        out['Psi'] = _replacePsiDiag(fallback['Psi'], sigma_rfx, mask_q)
    return out, RemlRefineMeta(valid=valid.detach(), clamped=clamped.detach())


def gateNormalRemlVsMap(
    current: dict[str, torch.Tensor],
    reml: dict[str, torch.Tensor],
    meta: RemlRefineMeta,
    n_total: torch.Tensor,
    mask_q: torch.Tensor | None = None,
    config: NormalRemlGateConfig = NormalRemlGateConfig(),
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return current-MAP stats with REML substituted on conservative rows."""
    q = reml['sigma_rfx_est'].shape[-1]
    if mask_q is None:
        active_q_count = torch.full_like(n_total, q, dtype=torch.long)
    else:
        active_q_count = mask_q[..., :q].bool().sum(dim=-1)
    use_reml = meta.valid & ~meta.clamped
    use_reml = use_reml & (active_q_count >= config.min_q)
    use_reml = use_reml & (n_total.to(device=use_reml.device) <= float(config.max_n_total))

    out = dict(current)
    for key in ['sigma_rfx_est', 'sigma_eps_est', 'Psi']:
        if key in current and key in reml:
            view = (slice(None),) + (None,) * (current[key].ndim - 1)
            out[key] = torch.where(use_reml[view], reml[key], current[key])
    return out, use_reml
