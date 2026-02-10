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
        root_idx = (roots_ * s).round().int().clamp_max(s-1)
        quantiles = sorted_samples[..., root_idx, :]
    else:
        raise NotImplementedError()
    return quantiles


