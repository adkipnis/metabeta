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


