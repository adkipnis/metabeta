"""Smoke test: cross-attention conditioning mode.

Verifies that 'concat' (baseline), 'cross_attn', and 'both' modes:
  1. Build without errors and produce consistent context dimensions.
  2. Run forward() (NLL) and backward() (sampling) without errors.
  3. Produce finite outputs.

Uses real small-n-mixed test data and the tiny / tiny-xattn / tiny-both model configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from metabeta.models.approximator import Approximator
from metabeta.utils.config import modelFromYaml
from metabeta.utils.dataloader import Dataloader

ROOT = Path(__file__).resolve().parents[2] / 'metabeta'
DATA = ROOT / 'outputs' / 'data' / 'small-n-mixed' / 'test.npz'
CFG_CONCAT = ROOT / 'configs' / 'models' / 'tiny.yaml'
CFG_XATTN = ROOT / 'configs' / 'models' / 'tiny-xattn.yaml'
CFG_BOTH = ROOT / 'configs' / 'models' / 'tiny-both.yaml'

if not DATA.exists():
    pytest.skip('small-n-mixed test data not found', allow_module_level=True)


@pytest.fixture(scope='module')
def batch() -> dict[str, torch.Tensor]:
    dl = Dataloader(DATA, batch_size=16)
    return next(iter(dl))


def _make_model(cfg_path: Path, batch: dict) -> Approximator:
    d_ffx = int(batch['X'].shape[-1])
    d_rfx = int(batch['Z'].shape[-1])
    cfg = modelFromYaml(cfg_path, d_ffx=d_ffx, d_rfx=d_rfx)
    return Approximator(cfg)


@pytest.mark.parametrize('cfg_path,label', [
    (CFG_CONCAT, 'concat'),
    (CFG_XATTN, 'cross_attn'),
    (CFG_BOTH, 'both'),
])
def test_forward_finite(cfg_path, label, batch):
    model = _make_model(cfg_path, batch)
    model.eval()
    with torch.no_grad():
        log_probs = model.forward(batch)
    for key in ('global', 'local'):
        lp = log_probs[key]
        n_bad = (~torch.isfinite(lp)).sum().item()
        assert n_bad == 0, f'[{label}] {key}: {n_bad}/{lp.numel()} non-finite log-probs'


@pytest.mark.parametrize('cfg_path,label', [
    (CFG_CONCAT, 'concat'),
    (CFG_XATTN, 'cross_attn'),
    (CFG_BOTH, 'both'),
])
def test_backward_finite(cfg_path, label, batch):
    model = _make_model(cfg_path, batch)
    model.eval()
    with torch.no_grad():
        proposal = model.estimate(batch, n_samples=8)
    for attr, name in [
        (proposal.samples_g, 'samples_g'),
        (proposal.log_prob_g, 'log_prob_g'),
    ]:
        n_bad = (~torch.isfinite(attr)).sum().item()
        assert n_bad == 0, f'[{label}] {name}: {n_bad}/{attr.numel()} non-finite values'


def test_xattn_context_dims(batch):
    """Cross-attn model context dims must be strictly smaller than concat."""
    model_c = _make_model(CFG_CONCAT, batch)
    model_x = _make_model(CFG_XATTN, batch)

    # In cross_attn mode, analytics leave the feature vector → summarizer_g input dim shrinks
    d_in_c = model_c.summarizer_g.proj_in[0].in_features
    d_in_x = model_x.summarizer_g.proj_in[0].in_features
    assert d_in_x < d_in_c, (
        f'cross_attn summarizer_g input ({d_in_x}) should be smaller than concat ({d_in_c})'
    )
    # context projectors must have the right input dims
    d_ctx_l = model_x.summarizer_l.proj_context.in_features
    d_ctx_g = model_x.summarizer_g.proj_context.in_features
    assert d_ctx_l == model_x._analyticsLocalDim(), (
        f'summarizer_l proj_context in_features ({d_ctx_l}) != _analyticsLocalDim ({model_x._analyticsLocalDim()})'
    )
    assert d_ctx_g == model_x._analyticsGlobalDim(), (
        f'summarizer_g proj_context in_features ({d_ctx_g}) != _analyticsGlobalDim ({model_x._analyticsGlobalDim()})'
    )
    # cross_attn summarizers must have cross_attn block; concat must not
    assert model_x.summarizer_l.cross_attn is not None, 'xattn: summarizer_l missing cross_attn block'
    assert model_x.summarizer_g.cross_attn is not None, 'xattn: summarizer_g missing cross_attn block'
    assert model_c.summarizer_l.cross_attn is None, 'concat: summarizer_l should not have cross_attn block'
    assert model_c.summarizer_g.cross_attn is None, 'concat: summarizer_g should not have cross_attn block'


def test_both_context_dims(batch):
    """Both mode: cross-attn context present AND feature dim equals concat (analytics not removed)."""
    model_c = _make_model(CFG_CONCAT, batch)
    model_b = _make_model(CFG_BOTH, batch)

    # In both mode, analytics stay in the feature vector → summarizer_g input dim == concat
    d_in_c = model_c.summarizer_g.proj_in[0].in_features
    d_in_b = model_b.summarizer_g.proj_in[0].in_features
    assert d_in_b == d_in_c, (
        f'both summarizer_g input ({d_in_b}) should equal concat ({d_in_c})'
    )
    # both mode must also have cross-attn blocks with correct context dims
    assert model_b.summarizer_l.cross_attn is not None, 'both: summarizer_l missing cross_attn block'
    assert model_b.summarizer_g.cross_attn is not None, 'both: summarizer_g missing cross_attn block'
    d_ctx_l = model_b.summarizer_l.proj_context.in_features
    d_ctx_g = model_b.summarizer_g.proj_context.in_features
    assert d_ctx_l == model_b._analyticsLocalDim(), (
        f'both summarizer_l proj_context ({d_ctx_l}) != _analyticsLocalDim ({model_b._analyticsLocalDim()})'
    )
    assert d_ctx_g == model_b._analyticsGlobalDim(), (
        f'both summarizer_g proj_context ({d_ctx_g}) != _analyticsGlobalDim ({model_b._analyticsGlobalDim()})'
    )
