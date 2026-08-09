import torch

from metabeta.utils.results import Proposal


def _proposal(d_corr: int = 0) -> Proposal:
    b, m, s, q, d = 3, 4, 5, 2, 6
    torch.manual_seed(0)
    return Proposal(
        {
            'global': {
                'samples': torch.randn(b, s, d + d_corr),
                'log_prob': torch.randn(b, s),
            },
            'local': {
                'samples': torch.randn(b, m, s, q),
                'log_prob': torch.randn(b, m, s),
            },
        },
        has_sigma_eps=True,
        d_corr=d_corr,
    )


def test_inplace_rescale_matches_allocating_rescale():
    scale = torch.rand(3) + 0.5

    allocating, in_place = _proposal(), _proposal()
    allocating.rescale(scale)
    in_place.rescale(scale, inplace=True)

    # bitwise, not approximate: both operands are float32, so the two paths run the same multiply
    assert torch.equal(in_place.samples_l, allocating.samples_l)
    assert torch.equal(in_place.samples_g, allocating.samples_g)


def test_inplace_rescale_leaves_correlations_untouched():
    scale = torch.rand(3) + 0.5
    d_corr = 3

    allocating, in_place = _proposal(d_corr), _proposal(d_corr)
    allocating.rescale(scale)
    in_place.rescale(scale, inplace=True)

    assert torch.equal(in_place.samples_g, allocating.samples_g)
    # the trailing correlation block is unitless and must survive rescaling unscaled
    assert torch.equal(
        in_place.samples_g[..., -d_corr:], _proposal(d_corr).samples_g[..., -d_corr:]
    )


def test_inplace_rescale_reuses_the_local_buffer():
    proposal = _proposal()
    before = proposal.samples_l
    proposal.rescale(torch.rand(3) + 0.5, inplace=True)

    # the point of the flag: no second copy of the largest array is allocated
    assert proposal.samples_l is before


def test_rescale_defaults_to_allocating_a_new_local_buffer():
    proposal = _proposal()
    before = proposal.samples_l
    original = before.clone()
    proposal.rescale(torch.rand(3) + 0.5)

    # default stays non-mutating, so callers sharing the buffer are unaffected
    assert proposal.samples_l is not before
    assert torch.equal(before, original)
