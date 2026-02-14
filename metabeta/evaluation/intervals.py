import torch
from metabeta.utils.evaluation import Proposal, getMasks


def getQuantiles(
    roots: tuple[float, float],
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    assert len(roots) == 2
    assert 0.0 <= roots[0] < roots[1] <= 1.0
    roots_ = torch.tensor(roots, device=samples.device)
    if weights is None:
        quantiles = torch.quantile(samples, roots_, dim=-2)
        quantiles = quantiles.movedim(0, -2)
    else:
        raise NotImplementedError()
    return quantiles # (b, ..., 2, d)


def sampleCredibleInterval(
    proposal: Proposal,
    alpha: float,
) -> dict[str, torch.Tensor]:
    out = {}
    roots = (alpha / 2, 1 - alpha / 2)
    w = None #proposal.weights
    ci_g = getQuantiles(roots, proposal.samples_g, w)
    out = proposal.partition(ci_g)
    out['rfx'] = getQuantiles(roots, proposal.samples_l, w)
    return out


def coverage(
    ci: torch.Tensor,  # credible interval
    gt: torch.Tensor,  # ground truth
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    above = ci[..., 0, :] - eps <= gt
    below = gt <= ci[..., 1, :] + eps
    inside = above & below
    if mask is not None:
        inside = inside & mask
        n = mask.sum(0).clamp_min(1.0)
        return inside.float().sum(0) / n
    else:
        return inside.float().mean(0)


def sampleCoverageError(
    ci_dict: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    nominal: float,
) -> dict[str, float]:
    out = {}
    masks = getMasks(data)
    for key, ci in ci_dict.items():
        gt = data[key]
        mask = masks[key]
        if key == 'sigma_eps':
            gt = gt.unsqueeze(-1)
            ci = ci.unsqueeze(-1)
        out[key] = coverage(ci, gt, mask) - nominal
        if key == 'rfx':   # average over groups
            out[key] = out[key].mean(0)
        out[key] = out[key].mean(0).item()   # TODO: incorporate mask?
    return out


def expectedCoverageError(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
    alphas: list[float] = [0.02, 0.05, 0.10, 0.20, 0.32, 0.50],
) -> dict[str, float]:
    # get coverage error for each alpha level
    ce_dict = {}
    for alpha in alphas:
        ci = sampleCredibleInterval(proposal, alpha=alpha)
        ce_dict[alpha] = sampleCoverageError(ci, data, nominal=(1 - alpha))

    # average over alphas
    keys = next(iter(ce_dict.values())).keys()
    out = {k: sum(d[k] for d in ce_dict.values()) / len(ce_dict) for k in keys}
    return out


