import torch

from metabeta.utils.posterior_cache import (
    loadProposalCache,
    posteriorSampleCacheName,
    saveProposalCache,
)
from metabeta.utils.results import Proposal


def test_posterior_sample_cache_name_includes_checkpoint_and_seed():
    name = posteriorSampleCacheName(
        partition='test',
        method='mb',
        checkpoint_name='data=toy_model=tiny_seed=13',
        checkpoint_prefix='best',
        n_samples=500,
        seed=7,
        k=2,
    )

    assert name == 'test.mb.data=toy_model=tiny_seed=13_best_s500_seed7_k2.npz'


def test_proposal_cache_round_trips_posterior_samples(tmp_path):
    proposal = Proposal(
        {
            'global': {
                'samples': torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5),
                'log_prob': torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3),
            },
            'local': {
                'samples': torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2),
                'log_prob': torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
            },
        },
        has_sigma_eps=True,
        d_corr=1,
        corr_rfx=torch.eye(2).expand(2, 3, 2, 2).clone(),
    )
    proposal.tpd = 0.25
    proposal.reff = 0.8
    proposal.is_results['weights'] = torch.full((2, 3), 1 / 3)

    path = tmp_path / 'test.mb.run_best_s3_seed0_k0.npz'
    saveProposalCache(path, proposal, metadata={'seed': 7, 'checkpoint_prefix': 'best'})
    loaded, metadata = loadProposalCache(path)

    torch.testing.assert_close(loaded.samples_g, proposal.samples_g)
    torch.testing.assert_close(loaded.samples_l, proposal.samples_l)
    torch.testing.assert_close(loaded.log_prob_g, proposal.log_prob_g)
    torch.testing.assert_close(loaded.log_prob_l, proposal.log_prob_l)
    torch.testing.assert_close(loaded.corr_rfx, proposal.corr_rfx)
    torch.testing.assert_close(loaded.weights, proposal.weights)
    assert loaded.has_sigma_eps is True
    assert loaded.d_corr == 1
    assert loaded.tpd == 0.25
    assert loaded.reff == 0.8
    assert metadata == {'seed': 7, 'checkpoint_prefix': 'best'}
