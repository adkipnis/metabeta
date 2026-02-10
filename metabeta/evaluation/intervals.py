import torch
from metabeta.utils.evaluation import Proposed, numFixed, getMasks


def sortProposed(proposed: Proposed) -> Proposed:
    for source in ('global', 'local'):
        samples = proposed[source]['samples']
        log_prob = proposed[source]['log_prob']
        sorted_samples, idx = samples.sort(dim=-2, descending=False)
        sorted_log_prob = torch.gather(log_prob, -1, idx[..., 0])
        proposed[source]['samples'] = sorted_samples
        proposed[source]['log_prob'] = sorted_log_prob
        # TODO: weights
    return proposed


def getQuantiles(
    roots: tuple[float, float],
    sorted_samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    assert len(roots) == 2
    assert 0.0 <= roots[0] < roots[1] <= 1.0
    s = sorted_samples.shape[-2]
    roots_ = torch.tensor(roots)
    if weights is None:
        root_idx = (roots_ * s).round().int().clamp_max(s - 1)
        quantiles = sorted_samples[..., root_idx, :]
    else:
        raise NotImplementedError()
    return quantiles


def sampleCredibleInterval(
    proposed: Proposed,
    alpha: float,
) -> dict[str, torch.Tensor]:
    proposed = sortProposed(proposed)
    d = numFixed(proposed)
    out = {}
    roots = (alpha / 2, 1 - alpha / 2)

    # global
    samples_g = proposed['global']['samples']
    weights_g = proposed['global'].get('weights')
    ci_g = getQuantiles(roots, samples_g, weights_g)
    out['ffx'] = ci_g[..., :d]
    out['sigma_rfx'] = ci_g[..., d:-1]
    out['sigma_eps'] = ci_g[..., -1]

    # local
    samples_l = proposed['local']['samples']
    weights_l = proposed['local'].get('weights')
    out['rfx'] = getQuantiles(roots, samples_l, weights_l)
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
    proposed: Proposed,
    data: dict[str, torch.Tensor],
    alphas: list[float] = [0.02, 0.05, 0.10, 0.20, 0.32, 0.50],
) -> dict[str, float]:
    # get coverage error for each alpha level
    ce_dict = {}
    for alpha in alphas:
        ci = sampleCredibleInterval(proposed, alpha=alpha)
        ce_dict[alpha] = sampleCoverageError(ci, data, nominal=(1 - alpha))

    # average over alphas
    keys = next(iter(ce_dict.values())).keys()
    out = {k: sum(d[k] for d in ce_dict.values()) / len(ce_dict) for k in keys}
    return out


