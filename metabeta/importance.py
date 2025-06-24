from typing import Dict
import torch
from torch import distributions as D


def logPriorBeta(beta: torch.Tensor, mask: torch.Tensor):
    # beta: (b, d, s)
    # mask: (b, d)
    lp = D.Normal(0., 3.).log_prob(beta)
    lp = lp * mask.unsqueeze(-1)
    return lp.sum(dim=1) # (b, s)


def logPrior(beta: torch.Tensor, sigma: torch.Tensor, mask: torch.Tensor):
    lp_beta = logPriorBeta(beta, mask)
    lp_sigma = D.HalfNormal(1.).log_prob(sigma.sqrt())
    return lp_beta + lp_sigma


def logLikelihood(beta: torch.Tensor, sigma: torch.Tensor, batch: dict):
    # y: (b, n)
    # X: (b, n, d)
    # beta: (b, d, s)
    # sigma: (b, 1) or (b, s)
    y, X, ns = batch['y'], batch['X'], batch['n'].unsqueeze(-1)

    # compute sum of squared residuals
    mu = torch.einsum('bnd,bds->bns', X, beta) # (b, n, s)
    ssr = (y - mu).square().sum(dim=1) # (b, s)

    # Compute log likelihood per batch
    ll = (
        - 0.5 * ns * torch.tensor(2 * torch.pi).log() 
        - ns * sigma.log()
        - 0.5 * ssr / sigma.square()
    )
    return ll # (b, s)


