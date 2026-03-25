from __future__ import annotations

from pathlib import Path
import pytest
import torch

from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.config import dataFromYaml, modelFromYaml


data_cfg_path = Path('metabeta', 'simulation', 'configs', 'toy.yaml')
data_fname = dataFromYaml(data_cfg_path, 'test')
DATA_PATH = Path('metabeta', 'outputs', 'data', data_fname)
if not DATA_PATH.exists():
    pytest.skip(
        'validation dataset not found',
        allow_module_level=True,
    )


@pytest.fixture(scope='module')
def batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    dl = Dataloader(DATA_PATH, batch_size=8)
    return next(iter(dl))


@pytest.fixture(scope='module')
def model() -> Approximator:
    model_cfg_path = Path('metabeta', 'models', 'configs', 'toy.yaml')
    model_cfg = modelFromYaml(model_cfg_path, 2, 1)
    return Approximator(model_cfg)


def test_forward_runs_and_is_finite(model: Approximator, batch: dict[str, torch.Tensor]):
    log_probs = model.forward(batch)
    for key in ('global', 'local'):
        assert key in log_probs, f'missing key {key}'
        assert torch.isfinite(log_probs[key]).all(), f'{key} contains non-finite values'


def test_backward_produces_gradients(model: Approximator, batch: dict[str, torch.Tensor]):
    log_probs = model.forward(batch)
    loss = (log_probs['global'].mean() + log_probs['local'].mean())
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), 'no gradients produced'
    assert all(g is None or torch.isfinite(g).all() for g in grads), 'non-finite gradients detected'


def test_estimate_shapes_and_constraints(model: Approximator, batch: dict[str, torch.Tensor]):
    model.forward(batch) # warmup
    model.eval()
    n_samples = 12
    proposal = model.estimate(batch, n_samples=n_samples)

    samples_g = proposal.samples_g
    logp_g = proposal.log_prob_g
    samples_l = proposal.samples_l
    logp_l = proposal.log_prob_l

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
