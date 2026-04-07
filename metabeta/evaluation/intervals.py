import torch
from metabeta.utils.evaluation import Proposal, getMasks, weightedQuantile
from metabeta.utils.regularization import corrToLower

EPS = 1e-6
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# --- Credible Intervals
def getQuantiles(
    roots: tuple[float, float],
    samples: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    assert len(roots) == 2
    assert 0.0 <= roots[0] < roots[1] <= 1.0
    roots_ = torch.tensor(roots, dtype=samples.dtype, device=samples.device)
    if weights is None:
        quantiles = torch.quantile(samples, roots_, dim=-2).movedim(0, -2)
    else:
        quantiles = weightedQuantile(samples, weights, roots_).movedim(-1, -2)
    return quantiles   # (b, ..., 2, d)


def getCredibleInterval(
    proposal: Proposal,
    alpha: float,
) -> dict[str, torch.Tensor]:
    # get equal-tailed credible interval from proposal posterior
    out = {}
    roots = (alpha / 2, 1 - alpha / 2)
    w = proposal.weights
    ci_g = getQuantiles(roots, proposal.samples_g, w)
    out = proposal.partition(ci_g)
    if proposal.d_corr > 0:
        out['corr_rfx'] = getQuantiles(roots, corrToLower(proposal.corr_rfx), w)
    out['rfx'] = getQuantiles(roots, proposal.samples_l, w)
    return out


def getCredibleIntervals(
    proposal: Proposal,
    alphas: list[float] = ALPHAS,
) -> dict[float, dict[str, torch.Tensor]]:
    return {alpha: getCredibleInterval(proposal, alpha) for alpha in alphas}


# --- Coverage
def getAtomicCoverage(
    ci: torch.Tensor,  # credible interval
    gt: torch.Tensor,  # ground truth
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # approximate the posterior coverage of the ground truth over batch
    above = ci[..., 0, :] - EPS <= gt
    below = gt <= ci[..., 1, :] + EPS
    inside = above & below
    if mask is not None:
        inside = inside & mask
        n = mask.sum(0).clamp_min(1.0)
        return inside.float().sum(0) / n
    return inside.float().mean(0)


def getCoveragePerParameter(
    ci_dict: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """get coverage for each parameter type"""
    out = {}
    masks = getMasks(data)
    for key, ci in ci_dict.items():
        if key == 'corr_rfx':
            out[key] = getAtomicCoverage(ci, corrToLower(data['corr_rfx']), mask=None)
            continue
        gt = data[key]
        mask = masks[key]
        if key == 'sigma_eps':
            gt = gt.unsqueeze(-1)
            ci = ci.unsqueeze(-1)
        out[key] = getAtomicCoverage(ci, gt, mask)
        if key == 'rfx':   # average over groups
            out[key] = out[key].mean(0)
    return out


def getCoverages(
    ci_dicts: dict[float, dict[str, torch.Tensor]],
    data: dict[str, torch.Tensor],
) -> dict[float, dict[str, torch.Tensor]]:
    out = {}
    for alpha, ci_per_parameter in ci_dicts.items():
        out[alpha] = getCoveragePerParameter(ci_per_parameter, data)
    return out


# --- Coverage Errors
def getCoverageError(
    coverages: dict[str, torch.Tensor],
    nominal: torch.Tensor,
    log_ratio: bool = True,
) -> dict[str, torch.Tensor]:
    """get coverage error for each parameter type"""
    # log(actual / nominal), sensitive to scale
    if log_ratio:
        return {k: cvrg.log() - nominal.log() for k, cvrg in coverages.items()}
    # plain coverage error, insensitive to scale
    return {k: cvrg - nominal for k, cvrg in coverages.items()}


def getCoverageErrors(
    cvrg_dicts: dict[float, dict[str, torch.Tensor]],
    log_ratio: bool = True,
) -> dict[float, dict[str, torch.Tensor]]:
    out = {}
    for alpha, observed in cvrg_dicts.items():
        nominal = torch.tensor(1 - alpha)
        out[alpha] = getCoverageError(observed, nominal, log_ratio=log_ratio)
    return out
