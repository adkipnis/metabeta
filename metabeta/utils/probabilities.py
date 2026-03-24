import math
import torch
from torch import distributions as D


def logPriorFfx(
    ffx: torch.Tensor,  # (b, s, d)
    nu_ffx: torch.Tensor,  # (b, 1, d)
    tau_ffx: torch.Tensor,  # (b, 1, d)
    mask: torch.Tensor | None = None,  # (b, 1, d)
) -> torch.Tensor:
    dist = D.Normal(nu_ffx, tau_ffx)
    lp = dist.log_prob(ffx)
    if mask is not None:
        lp = lp * mask
    return lp.sum(-1)   # (b, s)


def logPriorSigmaRfx(
    sigma_rfx: torch.Tensor,  # (b, s, q)
    tau_rfx: torch.Tensor,  # (b, 1, q)
    mask: torch.Tensor | None = None,  # (b, 1, q)
) -> torch.Tensor:
    dist = D.HalfNormal(scale=tau_rfx)
    lp = dist.log_prob(sigma_rfx)
    if mask is not None:
        lp = lp * mask
    return lp.sum(-1)   # (b, s)


def logPriorSigmaEps(
    sigma_eps: torch.Tensor,  # (b, s)
    tau_eps: torch.Tensor,  # (b, 1)
) -> torch.Tensor:
    dist = D.StudentT(df=4, loc=0, scale=tau_eps)
    lp = dist.log_prob(sigma_eps) + math.log(2.0)
    return lp   # (b, s)


def logPriorRfx(
    rfx: torch.Tensor,  # (b, m, s, q)
    sigma_rfx: torch.Tensor,  # (b, s, q)
    mask: torch.Tensor | None = None,  # (b, m, 1, q)
) -> torch.Tensor:
    scale = sigma_rfx.unsqueeze(1) + 1e-12
    dist = D.Normal(loc=0, scale=scale)
    lp = dist.log_prob(rfx)
    if mask is not None:
        lp = lp * mask
    return lp.sum(dim=(1, -1))   # (b, s)


def logLikelihoodCond(
    ffx: torch.Tensor,  # (b, s, d)
    sigma_eps: torch.Tensor,  # (b, s)
    rfx: torch.Tensor,  # (b, m, s, q)
    y: torch.Tensor,  # (b, m, n, 1)
    X: torch.Tensor,  # (b, m, n, d)
    Z: torch.Tensor,  # (b, m, n, q)
    mask: torch.Tensor | None = None,  # (b, m, n, 1)
) -> torch.Tensor:
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    loc = mu_g + mu_l
    scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
    dist = D.Normal(loc=loc, scale=scale)
    ll = dist.log_prob(y)
    if mask is not None:
        ll = ll * mask
    return ll.sum(dim=(1, 2))   # (b, s)
