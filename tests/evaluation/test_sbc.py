import torch

from metabeta.evaluation import sbc


def test_fractional_ranks_chunked_matches_unchunked(monkeypatch):
    samples = torch.randn(5, 3, 11, 2)
    targets = torch.randn(5, 3, 2)

    monkeypatch.setattr(sbc, '_RANK_CHUNK_ELEMS', 1_000_000)
    expected = sbc.fractionalRanks(samples, targets)

    monkeypatch.setattr(sbc, '_RANK_CHUNK_ELEMS', 30)
    actual = sbc.fractionalRanks(samples, targets)

    torch.testing.assert_close(actual, expected)


def test_fractional_ranks_chunked_matches_unchunked_with_weights(monkeypatch):
    samples = torch.randn(5, 3, 11, 2)
    targets = torch.randn(5, 3, 2)
    weights = torch.rand(5, 11)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    monkeypatch.setattr(sbc, '_RANK_CHUNK_ELEMS', 1_000_000)
    expected = sbc.fractionalRanks(samples, targets, weights)

    monkeypatch.setattr(sbc, '_RANK_CHUNK_ELEMS', 30)
    actual = sbc.fractionalRanks(samples, targets, weights)

    torch.testing.assert_close(actual, expected)
