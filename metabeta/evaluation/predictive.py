import torch
from torch import distributions as D
from metabeta.utils.evaluation import Proposal

def getPosteriorPredictive(
    proposal: Proposal,
    data: dict[str, torch.Tensor],
) -> D.Normal:
    ''' get p(y|X, theta) '''
    # parameters
    ffx = proposal.ffx # (b, s, d)
    sigma_eps = proposal.sigma_eps # (b, s)
    rfx = proposal.rfx # (b, m, s, q)

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
    pp: D.Normal,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = data['mask_n'].unsqueeze(-1)
    y = data['y'] # (b, m, n)
    nll = -pp.log_prob(y.unsqueeze(-1))
    nll = nll * mask
    nll = nll.sum(dim=(1,2)) / mask.sum(dim=(1,2))
    return nll # (b, s)


def posteriorPredictiveSample(
    pp: D.Normal,
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = data['mask_n'].unsqueeze(-1)
    y_rep = pp.sample((1,)).squeeze(0)
    y_rep = y_rep * mask
    return y_rep

