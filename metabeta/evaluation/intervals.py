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


