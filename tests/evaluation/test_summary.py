"""Tests for evaluation/summary.py."""

import torch

from metabeta.evaluation.summary import getSummary
from metabeta.utils.results import Proposal


def _twoDatasets() -> tuple[Proposal, dict]:
    """Two Normal datasets with 2 and 3 groups of 6 rows (d = 2, q = 1), carrying analytical
    statistics, and a proposal of 64 draws."""
    torch.manual_seed(0)
    B, m_pad, n, s = 2, 3, 6, 64
    mask_m = torch.tensor([[True, True, False], [True, True, True]])
    mask_n = mask_m[..., None].expand(B, m_pad, n)
    X = torch.cat([torch.ones(B, m_pad, n, 1), torch.randn(B, m_pad, n, 1)], -1) * mask_n[..., None]
    rfx = torch.randn(B, m_pad, 1) * mask_m[..., None]
    batch = {
        'X': X,
        'Z': X[..., :1],
        'y': (X.sum(-1) + rfx + 0.5 * torch.randn(B, m_pad, n)) * mask_n,
        'ffx': torch.ones(B, 2),
        'sigma_rfx': torch.ones(B, 1),
        'sigma_eps': torch.full((B,), 0.5),
        'rfx': rfx,
        'corr_rfx': torch.ones(B, 1, 1),
        'm': mask_m.sum(1),
        'ns': mask_n.sum(2),
        'mask_d': torch.ones(B, 2, dtype=torch.bool),
        'mask_q': torch.ones(B, 1, dtype=torch.bool),
        'mask_m': mask_m,
        'mask_mq': mask_m[..., None],
        'mask_n': mask_n,
        'nu_ffx': torch.zeros(B, 2),
        'tau_ffx': torch.ones(B, 2),
        'tau_rfx': torch.ones(B, 1),
        'tau_eps': torch.ones(B),
        'family_ffx': torch.zeros(B, dtype=torch.long),
        'family_sigma_rfx': torch.zeros(B, dtype=torch.long),
        'family_sigma_eps': torch.zeros(B, dtype=torch.long),
        'stats': {'blup_est': torch.randn(B, m_pad, 1) * mask_m[..., None]},
    }
    proposal = Proposal(
        {
            'global': {'samples': torch.randn(B, s, 4).abs(), 'log_prob': torch.zeros(B, s)},
            'local': {'samples': torch.randn(B, m_pad, s, 1), 'log_prob': torch.zeros(B, m_pad, s)},
        },
        has_sigma_eps=True,
    )
    return proposal, batch


def test_summary_chunks_slice_the_nested_statistics():
    """A batch with stored statistics summarizes one dataset at a time as it does in one
    chunk: each chunk takes its own rows of ``stats``."""
    proposal, batch = _twoDatasets()
    whole = getSummary(proposal, batch, dataset_chunk_size=16)
    single = getSummary(proposal, batch, dataset_chunk_size=1)
    assert torch.allclose(single.per_dataset.loo_nll, whole.per_dataset.loo_nll)
