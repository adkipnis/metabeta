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


def _twoNormalDatasets(s: int):
    """Two random-intercept Normal datasets with 3 and 5 groups of 8 rows, and a flow-like pool
    of s draws."""
    from metabeta.utils.results import Proposal

    g = torch.Generator().manual_seed(0)
    B, m, n = 2, 5, 8
    mask_m = torch.tensor([[True] * 3 + [False] * 2, [True] * 5])
    mask_n = mask_m[..., None].expand(B, m, n)
    ones = torch.ones(B, m, n, 1) * mask_n[..., None]
    rfx = 0.5 * torch.randn(B, m, 1, generator=g) * mask_m[..., None]
    batch = {
        'X': ones,
        'Z': ones,
        'y': (1.0 + rfx + 0.3 * torch.randn(B, m, n, generator=g)) * mask_n,
        'nu_ffx': torch.zeros(B, 1),
        'tau_ffx': torch.full((B, 1), 2.0),
        'tau_rfx': torch.ones(B, 1),
        'tau_eps': torch.ones(B),
        'eta_rfx': torch.zeros(B),
        'family_ffx': torch.zeros(B, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(B, dtype=torch.long),
        'family_sigma_eps': torch.zeros(B, dtype=torch.long),
        'mask_d': torch.ones(B, 1, dtype=torch.bool),
        'mask_q': torch.ones(B, 1, dtype=torch.bool),
        'mask_m': mask_m,
        'mask_mq': mask_m[..., None],
        'mask_n': mask_n,
    }
    samples = torch.cat(
        [
            1.0 + 0.3 * torch.randn(B, s, 1, generator=g),
            (0.5 + 0.1 * torch.randn(B, s, 2, generator=g)).abs(),
        ],
        -1,
    )
    pool = Proposal(
        {
            'global': {'samples': samples, 'log_prob': torch.zeros(B, s)},
            'local': {'samples': torch.zeros(B, m, s, 1), 'log_prob': torch.zeros(B, m, s)},
        },
        has_sigma_eps=True,
    )
    return pool, batch


def test_refined_proposal_carries_the_imh_diagnostics_through_chunks_and_cache(tmp_path) -> None:
    """Refined one dataset at a time, the IMH diagnostics are stored for every dataset and
    survive the posterior sample cache."""
    from metabeta.utils.posterior_cache import loadProposalCache, saveProposalCache
    from metabeta.utils.posterior_eval import refineProposal

    pool, batch = _twoNormalDatasets(4 * 60)
    torch.manual_seed(0)
    refined = refineProposal('imhMarginal', pool, batch, lf=0, batch_size=1)
    saveProposalCache(tmp_path / 'imh.npz', refined)
    cached, _ = loadProposalCache(tmp_path / 'imh.npz')
    keys = {'imh_accept_rate', 'imh_ess_min', 'imh_moved', 'imh_flag', 'imh_pool_k'}
    assert set(cached.is_results) == keys
    for k in keys:
        assert torch.equal(cached.is_results[k], refined.is_results[k].cpu())
        assert cached.is_results[k].shape == (2,)
    assert cached.is_results['imh_flag'].dtype == torch.bool
