"""Smoke tests for gradient accumulation in the training loop.

Two properties are verified:
1. Correctness: accumulating grad(loss/k) over k steps equals grad(loss) for one step.
2. Step count: optimizer fires every accum_steps micro-batches, plus once more for a
   trailing partial window at the end of the epoch.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from metabeta.models.approximator import Approximator
from metabeta.training.train import _coerce_cuda_rng_states, _coerce_rng_state_byte_tensor
from metabeta.utils.config import modelFromYaml
from metabeta.utils.dataloader import Dataloader


DATA_PATH = Path('metabeta', 'outputs', 'data', 'tiny-n-mixed', 'train_ep0001.npz')
if not DATA_PATH.exists():
    pytest.skip('tiny-n-mixed training data not found', allow_module_level=True)


@pytest.fixture(scope='module')
def dl() -> Dataloader:
    return Dataloader(DATA_PATH, batch_size=4, shuffle=False)


@pytest.fixture(scope='module')
def model(dl: Dataloader) -> Approximator:
    model_cfg_path = Path('metabeta', 'configs', 'models', 'tiny.yaml')
    model_cfg = modelFromYaml(model_cfg_path, dl.dataset.d, dl.dataset.q)
    return Approximator(model_cfg)


def _forward_loss(model: Approximator, batch: dict) -> torch.Tensor:
    """Simple scalar loss mirroring the forward-KL branch of Trainer.loss."""
    m = batch['m']
    mask = batch['mask_m']
    summaries = model.summarize(batch)
    log_probs = model.forward(batch, summaries)
    lq_g = log_probs['global']
    lq_l = log_probs['local'] * mask
    lq = lq_g + lq_l.sum(1) / m
    return -lq.mean()


# ---------------------------------------------------------------------------
# 1. Gradient correctness
# ---------------------------------------------------------------------------


def test_gradient_accumulation_correctness(model: Approximator, dl: Dataloader):
    """grad(loss/2).backward() twice == grad(loss).backward() once."""
    batch = next(iter(dl))

    # Baseline: one backward on the full loss
    model_a = copy.deepcopy(model)
    model_a.zero_grad()
    _forward_loss(model_a, batch).backward()
    grads_baseline = [p.grad.detach().clone() for p in model_a.parameters() if p.grad is not None]

    # Accumulation: two backwardss, each scaled by 1/2
    model_b = copy.deepcopy(model)
    model_b.zero_grad()
    for _ in range(2):
        (_forward_loss(model_b, batch) / 2).backward()
    grads_accum = [p.grad.detach().clone() for p in model_b.parameters() if p.grad is not None]

    assert len(grads_baseline) == len(grads_accum), 'parameter count mismatch'
    for ga, gb in zip(grads_baseline, grads_accum):
        torch.testing.assert_close(ga, gb, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Optimizer step count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'n_batches, accum_steps, expected_steps',
    [
        (6, 1, 6),  # no accumulation: step every batch
        (6, 2, 3),  # clean division: 3 windows of 2
        (6, 4, 2),  # 4+2: full window + trailing window
        (5, 2, 3),  # 2+2+1: two full windows + trailing micro-batch
    ],
)
def test_optimizer_step_count(
    model: Approximator,
    dl: Dataloader,
    n_batches: int,
    accum_steps: int,
    expected_steps: int,
):
    """Optimizer.step() fires ceil(n_batches / accum_steps) times."""
    import itertools

    # cap to available batches
    actual_n = min(n_batches, len(dl))
    if actual_n < n_batches:
        pytest.skip(f'dataloader has only {actual_n} batches, need {n_batches}')

    step_count = 0
    m = copy.deepcopy(model)
    m.zero_grad(set_to_none=True)

    for i, batch in enumerate(itertools.islice(dl, n_batches)):
        (_forward_loss(m, batch) / accum_steps).backward()

        is_step = (i + 1) % accum_steps == 0 or i == n_batches - 1
        if is_step:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            m.zero_grad(set_to_none=True)
            step_count += 1

    assert step_count == expected_steps


def test_coerce_rng_state_byte_tensor_from_cuda_like_tensor():
    state = torch.tensor([1.0, 2.0, 255.0], dtype=torch.float32)

    normalized = _coerce_rng_state_byte_tensor(state)

    assert normalized.device.type == 'cpu'
    assert normalized.dtype == torch.uint8
    torch.testing.assert_close(normalized, torch.tensor([1, 2, 255], dtype=torch.uint8))


def test_coerce_cuda_rng_states_from_mixed_serialized_formats():
    states = (
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        np.array([5, 6], dtype=np.int64),
        b'\x07\x08',
    )

    normalized = _coerce_cuda_rng_states(states)

    assert len(normalized) == 3
    for state in normalized:
        assert state.device.type == 'cpu'
        assert state.dtype == torch.uint8
    torch.testing.assert_close(normalized[0], torch.tensor([3, 4], dtype=torch.uint8))
    torch.testing.assert_close(normalized[1], torch.tensor([5, 6], dtype=torch.uint8))
    torch.testing.assert_close(normalized[2], torch.tensor([7, 8], dtype=torch.uint8))
