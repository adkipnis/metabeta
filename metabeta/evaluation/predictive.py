import torch
from torch import distributions as D
from metabeta.utils.evaluation import Proposed, numFixed

def samplePosteriorPredictive(
    proposed: Proposed,
    data: dict[str, torch.Tensor],
) -> D.Normal:
    ''' get p(y|X, theta) '''
    d = numFixed(proposed)

    # parameters
    ffx = proposed['global']['samples'][..., :d] # (b, s, d)
    sigma_eps = proposed['global']['samples'][..., -1] # (b, s)
    rfx = proposed['local']['samples'] # (b, m, s, q)

    # observations
    X = data['X'] # (b, m, n, d)
    Z = data['Z'] # (b, m, n, q)

    # posterior params
    mu_g = torch.einsum('bmnd,bsd->bmns', X, ffx)
    mu_l = torch.einsum('bmnq,bmsq->bmns', Z, rfx)
    loc = mu_g + mu_l
    scale = sigma_eps.unsqueeze(1).unsqueeze(1) + 1e-12
    return D.Normal(loc=loc, scale=scale)


def posteriorPredictiveNLL(
