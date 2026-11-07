import torch

from metabeta.evaluation.point import getPointEstimates
from metabeta.utils.posterior_eval import fit2proposal


def test_fit2proposal_random_intercept_has_no_correlations() -> None:
    # fit.py stores a (S, 1, 1) corr_rfx for a random intercept, which has no correlations
    B, S, d, m = 3, 8, 2, 4
    batch = {
        'nuts2_ffx': torch.randn(B, S, d),
        'nuts2_sigma_rfx': torch.rand(B, S, 1),
        'nuts2_rfx': torch.randn(B, m, S, 1),
        'nuts2_corr_rfx': torch.ones(B, S, 1, 1),
        'nuts2_duration': torch.ones(B),
    }
    proposal = fit2proposal(batch, 'nuts2')
    assert proposal.corr_rfx is None
    assert 'corr_rfx' not in getPointEstimates(proposal, 'mean')
