from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from metabeta.evaluation.predictive import (
    applyObsMask,
    estimateOOSProposals,
    getPosteriorPredictive,
    makeOOSSplits,
    oosNLL,
    posteriorPredictiveNLL,
)
from metabeta.models.approximator import Approximator
from metabeta.utils.config import dataFromYaml, modelFromYaml
from metabeta.utils.dataloader import Dataloader


# ---------------------------------------------------------------------------
# Dataset fixture (skipped when data is absent)
# ---------------------------------------------------------------------------

data_cfg_path = Path('metabeta', 'simulation', 'configs', 'toy-n.yaml')
data_fname = dataFromYaml(data_cfg_path, 'test')
DATA_PATH = Path('metabeta', 'outputs', 'data', data_fname)
if not DATA_PATH.exists():
    pytest.skip('toy test dataset not found', allow_module_level=True)


@pytest.fixture(scope='module')
def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    dl = Dataloader(DATA_PATH, batch_size=8)
    return next(iter(dl))


@pytest.fixture(scope='module')
def model() -> Approximator:
    model_cfg_path = Path('metabeta', 'models', 'configs', 'toy.yaml')
    model_cfg = modelFromYaml(model_cfg_path, d_ffx=2, d_rfx=1)
    m = Approximator(model_cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_data(B: int = 4, m: int = 5, n: int = 10) -> dict[str, torch.Tensor]:
    """Synthetic data dict with just the fields applyObsMask touches."""
    mask_n = torch.ones(B, m, n, dtype=torch.bool)
    # make the last two columns per group padding to simulate variable ns
    mask_n[:, :, -2:] = False
    ns = mask_n.sum(dim=-1).to(torch.int64)
    return {
        'mask_n': mask_n,
        'ns': ns,
        'n': ns.sum(dim=-1).to(torch.int32),
        'mask_m': (ns > 0),
        'y': torch.randn(B, m, n),
    }


# ---------------------------------------------------------------------------
# makeOOSSplits
# ---------------------------------------------------------------------------


class TestMakeOOSSplits:
    def test_returns_correct_number_of_splits(self):
        mask_n = torch.ones(2, 3, 8, dtype=torch.bool)
        splits = makeOOSSplits(mask_n, n_splits=5)
        assert len(splits) == 5

    def test_masks_are_bool_and_correct_shape(self):
        B, m, n = 4, 6, 12
        mask_n = torch.ones(B, m, n, dtype=torch.bool)
        for train, test in makeOOSSplits(mask_n, n_splits=3):
            assert train.shape == (B, m, n)
            assert test.shape == (B, m, n)
            assert train.dtype == torch.bool
            assert test.dtype == torch.bool

    def test_partition_of_mask_n(self):
        """train | test == mask_n and train & test is empty."""
        torch.manual_seed(1)
        B, m, n = 4, 5, 15
        mask_n = torch.ones(B, m, n, dtype=torch.bool)
        mask_n[:, :, -3:] = False  # some padding

        g = torch.Generator()
        g.manual_seed(42)
        for train, test in makeOOSSplits(mask_n, n_splits=4, generator=g):
            assert (train & test).any() == False, 'train and test overlap'
            assert ((train | test) == mask_n).all(), 'train | test != mask_n'

    def test_padding_never_assigned(self):
        """Positions outside mask_n must remain False in both masks."""
        torch.manual_seed(2)
        B, m, n = 3, 4, 10
        mask_n = torch.zeros(B, m, n, dtype=torch.bool)
        mask_n[:, :, :7] = True  # only first 7 obs per group are valid

        for train, test in makeOOSSplits(mask_n, n_splits=3):
            assert (~train[:, :, 7:]).all(), 'padding crept into train mask'
            assert (~test[:, :, 7:]).all(), 'padding crept into test mask'

    def test_test_fraction_approximately_p_test(self):
        """Mean test fraction across valid obs should be close to p_test."""
        torch.manual_seed(3)
        mask_n = torch.ones(16, 20, 25, dtype=torch.bool)
        p_test = 0.3
        g = torch.Generator()
        g.manual_seed(99)
        splits = makeOOSSplits(mask_n, n_splits=10, p_test=p_test, generator=g)
        fracs = [test.float().sum() / mask_n.float().sum() for _, test in splits]
        mean_frac = torch.tensor(fracs).mean().item()
        assert abs(mean_frac - p_test) < 0.05, f'mean test fraction {mean_frac:.3f} far from {p_test}'

    def test_every_group_has_at_least_one_train_obs(self):
        """No group with valid observations should be left with zero train obs."""
        torch.manual_seed(4)
        B, m, n = 6, 8, 5
        mask_n = torch.ones(B, m, n, dtype=torch.bool)
        # include single-observation groups
        mask_n[:, 0, 1:] = False  # group 0: only 1 obs per dataset

        g = torch.Generator()
        g.manual_seed(7)
        # high p_test to stress the starvation guard
        for train, test in makeOOSSplits(mask_n, n_splits=10, p_test=0.9, generator=g):
            has_data = mask_n.any(dim=-1)          # (B, m)
            train_count = train.sum(dim=-1)         # (B, m)
            starved = has_data & (train_count == 0)
            assert not starved.any(), 'some group with data has zero train observations'

    def test_reproducibility_with_generator(self):
        """Same generator seed should produce identical splits."""
        mask_n = torch.ones(3, 4, 10, dtype=torch.bool)
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        splits1 = makeOOSSplits(mask_n, n_splits=3, generator=g1)
        splits2 = makeOOSSplits(mask_n, n_splits=3, generator=g2)
        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            assert (tr1 == tr2).all()
            assert (te1 == te2).all()

    def test_different_seeds_produce_different_splits(self):
        mask_n = torch.ones(4, 6, 20, dtype=torch.bool)
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(1)
        [(tr1, _)] = makeOOSSplits(mask_n, n_splits=1, generator=g1)
        [(tr2, _)] = makeOOSSplits(mask_n, n_splits=1, generator=g2)
        assert not (tr1 == tr2).all(), 'different seeds produced identical splits'


# ---------------------------------------------------------------------------
# applyObsMask
# ---------------------------------------------------------------------------


class TestApplyObsMask:
    def test_mask_n_replaced(self):
        data = _minimal_data()
        new_mask = data['mask_n'].clone()
        new_mask[:, :, :2] = False
        out = applyObsMask(data, new_mask)
        assert (out['mask_n'] == new_mask).all()

    def test_ns_recomputed(self):
        data = _minimal_data()
        new_mask = data['mask_n'].clone()
        new_mask[:, :, :3] = False  # remove 3 obs per group
        out = applyObsMask(data, new_mask)
        expected_ns = new_mask.sum(dim=-1).to(data['ns'].dtype)
        assert (out['ns'] == expected_ns).all()

    def test_n_recomputed(self):
        data = _minimal_data()
        new_mask = data['mask_n'].clone()
        new_mask[:, 0, :] = False  # zero out first group
        out = applyObsMask(data, new_mask)
        expected_n = new_mask.sum(dim=(1, 2)).to(data['n'].dtype)
        assert (out['n'] == expected_n).all()

    def test_mask_m_follows_ns(self):
        data = _minimal_data()
        new_mask = data['mask_n'].clone()
        new_mask[:, 2, :] = False  # group 2 now has zero obs
        out = applyObsMask(data, new_mask)
        expected_mask_m = out['ns'] > 0
        assert (out['mask_m'] == expected_mask_m).all()
        assert not out['mask_m'][:, 2].any(), 'zeroed group should not be in mask_m'

    def test_y_not_copied(self):
        """y (and other unrelated fields) should be the same tensor object."""
        data = _minimal_data()
        new_mask = data['mask_n'].clone()
        out = applyObsMask(data, new_mask)
        assert out['y'] is data['y']

    def test_original_data_not_mutated(self):
        data = _minimal_data()
        original_ns = data['ns'].clone()
        new_mask = data['mask_n'].clone()
        new_mask[:, :, :4] = False
        applyObsMask(data, new_mask)
        assert (data['ns'] == original_ns).all(), 'original data was mutated'

    def test_ns_dtype_preserved(self):
        data = _minimal_data()
        out = applyObsMask(data, data['mask_n'])
        assert out['ns'].dtype == data['ns'].dtype

    def test_n_dtype_preserved(self):
        data = _minimal_data()
        out = applyObsMask(data, data['mask_n'])
        assert out['n'].dtype == data['n'].dtype


# ---------------------------------------------------------------------------
# estimateOOSProposals
# ---------------------------------------------------------------------------


class TestEstimateOOSProposals:
    def test_returns_one_proposal_per_split(self, batch, model):
        train_masks = [t for t, _ in makeOOSSplits(batch['mask_n'], n_splits=3)]
        proposals = estimateOOSProposals(model, batch, train_masks, n_samples=8)
        assert len(proposals) == 3

    def test_proposal_global_shape(self, batch, model):
        n_samples = 10
        train_masks = [t for t, _ in makeOOSSplits(batch['mask_n'], n_splits=2)]
        proposals = estimateOOSProposals(model, batch, train_masks, n_samples=n_samples)
        B = batch['y'].shape[0]
        for p in proposals:
            assert p.samples_g.shape[0] == B
            assert p.samples_g.shape[1] == n_samples

    def test_proposal_local_shape(self, batch, model):
        n_samples = 10
        train_masks = [t for t, _ in makeOOSSplits(batch['mask_n'], n_splits=2)]
        proposals = estimateOOSProposals(model, batch, train_masks, n_samples=n_samples)
        B = batch['y'].shape[0]
        m = batch['mask_m'].shape[1]
        for p in proposals:
            assert p.samples_l.shape == (B, m, n_samples, 1)

    def test_train_only_obs_used(self, batch, model):
        """Proposals from disjoint train splits should differ."""
        g = torch.Generator().manual_seed(0)
        splits = makeOOSSplits(batch['mask_n'], n_splits=2, p_test=0.5, generator=g)
        train_masks = [t for t, _ in splits]
        proposals = estimateOOSProposals(model, batch, train_masks, n_samples=16)
        means = [p.samples_g.mean().item() for p in proposals]
        assert means[0] != means[1], 'proposals from different splits are identical'

    def test_rng_reproducible(self, batch, model):
        """Same Generator seed gives identical proposals."""
        masks = [t for t, _ in makeOOSSplits(batch['mask_n'], n_splits=1,
                                              generator=torch.Generator().manual_seed(0))]
        p1 = estimateOOSProposals(model, batch, masks, n_samples=16, rng=np.random.default_rng(7))[0]
        p2 = estimateOOSProposals(model, batch, masks, n_samples=16, rng=np.random.default_rng(7))[0]
        assert torch.allclose(p1.samples_g, p2.samples_g), 'same Generator seed gave different global samples'
        assert torch.allclose(p1.samples_l, p2.samples_l), 'same Generator seed gave different local samples'

    def test_rng_generator_advances_across_splits(self, batch, model):
        """A single Generator shared across splits produces different samples per split."""
        masks = [t for t, _ in makeOOSSplits(batch['mask_n'], n_splits=2,
                                              generator=torch.Generator().manual_seed(1))]
        proposals = estimateOOSProposals(model, batch, masks, n_samples=16,
                                         rng=np.random.default_rng(5))
        assert not torch.allclose(proposals[0].samples_g, proposals[1].samples_g), (
            'Generator did not advance across splits — both splits got identical base samples'
        )

    def test_estimate_rng_reproducible(self, batch, model):
        """model.estimate with same Generator seed produces identical samples."""
        p1 = model.estimate(batch, n_samples=16, rng=np.random.default_rng(99))
        p2 = model.estimate(batch, n_samples=16, rng=np.random.default_rng(99))
        assert torch.allclose(p1.samples_g, p2.samples_g), 'same Generator seed gave different global samples'
        assert torch.allclose(p1.samples_l, p2.samples_l), 'same Generator seed gave different local samples'

    def test_estimate_rng_global_local_differ(self, batch, model):
        """With a shared Generator, global and local posteriors draw distinct samples."""
        p = model.estimate(batch, n_samples=16, rng=np.random.default_rng(3))
        # If both flows used a fresh RandomState from the same seed they'd produce
        # identical initial draws; a shared advancing Generator prevents this.
        global_flat = p.samples_g.reshape(-1)
        local_flat = p.samples_l.reshape(-1)[:global_flat.numel()]
        assert not torch.allclose(global_flat, local_flat), (
            'global and local samples appear identical — rng state is not advancing between them'
        )


# ---------------------------------------------------------------------------
# oosNLL (integration)
# ---------------------------------------------------------------------------


class TestOosNLL:
    def test_output_shape(self, batch, model):
        B = batch['y'].shape[0]
        result = oosNLL(model, batch, n_splits=2, n_samples=16)
        assert result.shape == (B,)

    def test_output_finite(self, batch, model):
        result = oosNLL(model, batch, n_splits=2, n_samples=16)
        assert torch.isfinite(result).all(), 'OOS NLL contains non-finite values'

    def test_averaging_over_splits(self, batch, model):
        """n_splits=1 and n_splits=3 should give different values (different random draws)."""
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        r1 = oosNLL(model, batch, n_splits=1, n_samples=16, generator=g1)
        r3 = oosNLL(model, batch, n_splits=3, n_samples=16, generator=g2)
        # they use different numbers of splits → different random masks → different means
        assert not torch.allclose(r1, r3), 'single-split and 3-split results are identical'

    def test_reproducible_with_rng(self, batch, model):
        """Same Generator seed gives identical OOS NLL."""
        g1 = torch.Generator().manual_seed(11)
        g2 = torch.Generator().manual_seed(11)
        r1 = oosNLL(model, batch, n_splits=2, n_samples=16, generator=g1, rng=np.random.default_rng(42))
        r2 = oosNLL(model, batch, n_splits=2, n_samples=16, generator=g2, rng=np.random.default_rng(42))
        assert torch.allclose(r1, r2), 'same Generator seed should give identical OOS NLL'

    def test_different_rngs_give_different_results(self, batch, model):
        """Different Generator seeds should produce different NLL values."""
        g1 = torch.Generator().manual_seed(11)
        g2 = torch.Generator().manual_seed(11)
        r1 = oosNLL(model, batch, n_splits=2, n_samples=16, generator=g1, rng=np.random.default_rng(42))
        r2 = oosNLL(model, batch, n_splits=2, n_samples=16, generator=g2, rng=np.random.default_rng(99))
        assert not torch.allclose(r1, r2), 'different Generator seeds gave identical OOS NLL'

    def test_mixture_and_expected_modes_differ(self, batch, model):
        """mixture and expected modes score the same draws differently."""
        g1 = torch.Generator().manual_seed(55)
        g2 = torch.Generator().manual_seed(55)
        r_mix = oosNLL(model, batch, n_splits=2, n_samples=32, mode='mixture', generator=g1, rng=np.random.default_rng(7))
        r_exp = oosNLL(model, batch, n_splits=2, n_samples=32, mode='expected', generator=g2, rng=np.random.default_rng(7))
        # Both should be finite, and the two modes should give different values
        assert torch.isfinite(r_mix).all()
        assert torch.isfinite(r_exp).all()
        assert not torch.allclose(r_mix, r_exp), 'mixture and expected modes gave identical NLL'

    def test_oos_nll_uses_test_obs_not_train(self, batch, model):
        """Sanity-check: scoring on train mask should differ from scoring on test mask."""
        g = torch.Generator().manual_seed(5)
        [(train_mask, test_mask)] = makeOOSSplits(
            batch['mask_n'], n_splits=1, p_test=0.3, generator=g
        )
        proposals = estimateOOSProposals(model, batch, [train_mask], n_samples=32)
        proposal = proposals[0]
        pp = getPosteriorPredictive(proposal, batch, likelihood_family=0)

        data_train = applyObsMask(batch, train_mask)
        data_test = applyObsMask(batch, test_mask)
        nll_train = posteriorPredictiveNLL(pp, data_train)
        nll_test = posteriorPredictiveNLL(pp, data_test)

        # NLL values should differ since they score different observations
        assert not torch.allclose(nll_train, nll_test), (
            'train and test NLL are identical — test masking is not working'
        )

    def test_expected_mode_runs(self, batch, model):
        result = oosNLL(model, batch, n_splits=2, n_samples=16, mode='expected')
        assert result.shape == (batch['y'].shape[0],)
        assert torch.isfinite(result).all()
