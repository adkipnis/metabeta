from __future__ import annotations

from pathlib import Path
import pytest
import torch

from metabeta.models.approximator import (
    Approximator,
    ApproximatorConfig,
    PosteriorConfig,
    SummarizerConfig,
)
from metabeta.utils.dataloader import Dataloader

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'metabeta' / 'outputs' / 'data' / 'valid_d3_q1_m5-30_n10-70_toy.npz'
if not DATA_PATH.exists():
    pytest.skip(
        'validation dataset val_d3_q1_m5-30_n10-70_toy.npz not found',
        allow_module_level=True,
    )


@pytest.fixture(scope='module')
def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    dl = Dataloader(DATA_PATH, batch_size=8)
    return next(iter(dl))


@pytest.fixture(scope='module')
def model() -> Approximator:
    s_cfg = SummarizerConfig(
        d_model=16,
        d_ff=32,
        d_output=16,
        n_blocks=2,
        n_isab=0,
        activation='GELU',
        dropout=0.0,
    )
    p_cfg = PosteriorConfig(
        transform='affine',
        subnet_kwargs={'activation': 'GELU', 'zero_init': True},
        n_blocks=2,
    )
    cfg = ApproximatorConfig(
        d_ffx=3,
        d_rfx=1,
        summarizer=s_cfg,
        posterior=p_cfg,
    )
    return Approximator(cfg)


def test_forward_runs_and_is_finite(model: Approximator, batch: dict[str, torch.Tensor]):
    loss = model.forward(batch)
    assert torch.isfinite(loss).all(), 'loss contains non-finite values'


def test_backward_produces_gradients(model: Approximator, batch: dict[str, torch.Tensor]):
    loss = model.forward(batch).mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), 'no gradients produced'
    assert all(g is None or torch.isfinite(g).all() for g in grads), 'non-finite gradients detected'


def test_estimate_shapes_and_constraints(model: Approximator, batch: dict[str, torch.Tensor]):
    model.forward(batch) # warmup
    model.eval()
    n_samples = 12
    proposed = model.estimate(batch, n_samples=n_samples)

    assert 'global' in proposed
    assert 'local' in proposed

    samples_g = proposed['global']['samples']
    logp_g = proposed['global']['log_prob']
    samples_l = proposed['local']['samples']
    logp_l = proposed['local']['log_prob']

    b = batch['y'].shape[0]
    m = batch['mask_m'].shape[1]

    assert samples_g.shape[0] == b
    assert samples_g.shape[1] == n_samples
    assert logp_g.shape == (b, n_samples)

    assert samples_l.shape == (b, m, n_samples, 1)
    assert logp_l.shape == (b, m, n_samples)

    d = model.d_ffx
    sigmas = samples_g[..., d:]
    assert (sigmas > 0).all(), 'global sigmas must be positive'
