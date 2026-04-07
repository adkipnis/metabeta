from __future__ import annotations

from pathlib import Path
import pytest
import torch

from metabeta.models.approximator import Approximator
from metabeta.utils.dataloader import Dataloader
from metabeta.utils.config import modelFromYaml


DATA_PATH = Path('metabeta', 'outputs', 'data', 'tiny-n-toy', 'test.npz')
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
    dl = Dataloader(DATA_PATH, batch_size=8)
    model_cfg_path = Path('metabeta', 'configs', 'models', 'tiny.yaml')
    model_cfg = modelFromYaml(model_cfg_path, dl.dataset.d, dl.dataset.q)
    return Approximator(model_cfg)


def test_forward_runs_and_is_finite(model: Approximator, batch: dict[str, torch.Tensor]):
    log_probs = model.forward(batch)
    for key in ('global', 'local'):
        assert key in log_probs, f'missing key {key}'
        assert torch.isfinite(log_probs[key]).all(), f'{key} contains non-finite values'


def test_backward_produces_gradients(model: Approximator, batch: dict[str, torch.Tensor]):
    log_probs = model.forward(batch)
    loss = log_probs['global'].mean() + log_probs['local'].mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), 'no gradients produced'
    assert all(g is None or torch.isfinite(g).all() for g in grads), 'non-finite gradients detected'


def test_estimate_shapes_and_constraints(model: Approximator, batch: dict[str, torch.Tensor]):
    model.forward(batch)   # warmup
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

    assert samples_l.shape == (b, m, n_samples, model.d_rfx)
    assert logp_l.shape == (b, m, n_samples)

    d = model.d_ffx
    q_var = model.d_rfx + (1 if model.has_sigma_eps else 0)
    sigmas = samples_g[..., d : d + q_var]
    mask_g = model._masks(batch, local=False)
    sigma_mask = mask_g[..., d : d + q_var].unsqueeze(1).expand_as(sigmas)
    assert (sigmas[sigma_mask] > 0).all(), 'unmasked global sigmas must be positive'
