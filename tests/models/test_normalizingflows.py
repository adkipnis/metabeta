from __future__ import annotations

import pytest
import torch

from metabeta.models.normalizingflows import ActNorm, LU, Permute
from metabeta.models.normalizingflows.coupling import Coupling, DualCoupling, CouplingFlow

ATOL = 5e-4
B = 32

# -----------------------
# Fixtures
# -----------------------


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


@pytest.fixture
def batch(_seed_rng):
    return torch.randn(B, 10, 3)


@pytest.fixture
def context(_seed_rng):
    return torch.randn(B, 10, 5)


@pytest.fixture(
    params=[
        ('affine', {}),
        ('spline', {}),
        ('spline', {'adaptive_tails': True}),
        ('spline', {'adaptive_domain': True, 'adaptive_tails': True}),
    ],
    ids=['affine', 'spline_fixed', 'spline_fixed_tails', 'spline_adaptive'],
)
def transform_cfg(request: pytest.FixtureRequest) -> tuple[str, dict]:
    return request.param


@pytest.fixture(params=['mlp', 'residual'])
def net_type(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=[True, False])
def zero_init(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture
def subnet_kwargs(net_type: str, zero_init: bool) -> dict[str, object]:
    return {
        'net_type': net_type,
        'zero_init': zero_init,
    }


def random_mask(x: torch.Tensor):
    mask = torch.randint(low=0, high=2, size=x.shape, dtype=torch.float32)
    x = x * mask
    return x, mask


def _numerical_logdet(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample log|det J| by explicitly building the Jacobian.
    """
    assert x.requires_grad
    jac_rows = []
    for i in range(z.shape[-1]):
        grad_z_i = torch.autograd.grad(z[..., i].sum(), x, retain_graph=True)[0]
        jac_rows.append(grad_z_i)
    jac = torch.stack(jac_rows, dim=-1)
    return torch.log(torch.abs(torch.det(jac)))


# -----------------------
# ActNorm
# -----------------------
def test_actnorm(batch):
    model = ActNorm(batch.shape[-1])
    x, mask = random_mask(batch)
    z, _, _ = model.forward(x, mask=mask)
    x_rec, _, _ = model.inverse(z, mask=mask)
    assert torch.isfinite(z).all(), 'anomaly in forward pass'
    assert torch.isfinite(x_rec).all(), 'anomaly in backward pass'
    assert torch.allclose(x, x_rec, atol=ATOL), 'recovery failed'
    x = x.detach().clone().requires_grad_(True)
    z, log_det, _ = model.forward(x, mask=mask)
    log_det_num = _numerical_logdet(z, x)
    assert torch.allclose(
        log_det, log_det_num, atol=ATOL
    ), f'Log determinant mismatch! Computed: {log_det}, Numerical: {log_det_num}'


# -----------------------
# LU
# -----------------------
@pytest.mark.parametrize('identity_init', [False, True])
def test_lu(batch, identity_init):
    model = LU(batch.shape[-1], identity_init=identity_init)
    model.eval()
    x, mask = random_mask(batch)
    z, _, _ = model.forward(x, mask=mask)
    x_rec, _, _ = model.inverse(z, mask=mask)
    assert torch.isfinite(z).all(), 'anomaly in forward pass'
    assert torch.isfinite(x_rec).all(), 'anomaly in backward pass'
    assert torch.allclose(x, x_rec, atol=ATOL), 'recovery failed'
    x = x.detach().clone().requires_grad_(True)
    z, log_det, _ = model.forward(x, mask=mask)
    log_det_num = _numerical_logdet(z, x)
    assert torch.allclose(
        log_det, log_det_num, atol=ATOL
    ), f'Log determinant mismatch! Computed: {log_det}, Numerical: {log_det_num}'


# -----------------------
# Permute
# -----------------------
def test_permute(batch):
    model = Permute(batch.shape[-1])
    x, mask = random_mask(batch)
    z, _, _ = model.forward(x, mask=mask)
    x_rec, _, _ = model.inverse(z, mask=mask)
    assert torch.isfinite(z).all(), 'anomaly in forward pass'
    assert torch.isfinite(x_rec).all(), 'anomaly in backward pass'
    assert torch.allclose(x, x_rec, atol=ATOL), 'recovery failed'
    x = x.detach().clone().requires_grad_(True)
    z, log_det, _ = model.forward(x, mask=mask)
    log_det_num = _numerical_logdet(z, x)
    assert torch.allclose(
        log_det, log_det_num, atol=ATOL
    ), f'Log determinant mismatch! Computed: {log_det}, Numerical: {log_det_num}'


# -----------------------
# Single Coupling
# -----------------------
def test_single_coupling(batch, context, subnet_kwargs, transform_cfg):
    transform, rq_kwargs = transform_cfg
    x, mask = random_mask(batch)
    x1, x2 = x.chunk(2, dim=-1)
    mask1, mask2 = mask.chunk(2, dim=-1)
    split_dims = (x1.shape[-1], x2.shape[-1])
    model = Coupling(
        split_dims=split_dims,
        d_context=context.shape[-1],
        subnet_kwargs=subnet_kwargs,
        transform=transform,
        rq_kwargs=rq_kwargs,
    )
    model.eval()
    (z1, z2), _ = model.forward(x1, x2, context=context, mask1=mask1, mask2=mask2)
    (x1_rec, x2_rec), _ = model.inverse(z1, z2, context=context, mask1=mask1, mask2=mask2)
    assert torch.isfinite(z1).all(), 'anomaly in left forward pass'
    assert torch.isfinite(z2).all(), 'anomaly in right forward pass'
    assert torch.isfinite(x1_rec).all(), 'anomaly in left backward pass'
    assert torch.isfinite(x2_rec).all(), 'anomaly in right backward pass'
    assert torch.allclose(x1, x1_rec, atol=ATOL), 'left recovery failed'
    assert torch.allclose(x2, x2_rec, atol=ATOL), 'right recovery failed'


# -----------------------
# Dual Coupling
# -----------------------
def test_dual_coupling(batch, context, subnet_kwargs, transform_cfg):
    transform, rq_kwargs = transform_cfg
    x, mask = random_mask(batch)
    model = DualCoupling(
        d_target=x.shape[-1],
        d_context=context.shape[-1],
        subnet_kwargs=subnet_kwargs,
        transform=transform,
        rq_kwargs=rq_kwargs,
    )
    model.eval()
    z, _, _ = model.forward(x, context=context, mask=mask)
    x_rec, _, _ = model.inverse(z, context, mask=mask)
    assert torch.isfinite(z).all(), 'anomaly in forward pass'
    assert torch.isfinite(x_rec).all(), 'anomaly in backward pass'
    assert torch.allclose(x, x_rec, atol=ATOL), 'recovery failed'
    x = x.detach().clone().requires_grad_(True)
    z, log_det, _ = model.forward(x, context=context, mask=mask)
    log_det_num = _numerical_logdet(z, x)
    assert torch.allclose(
        log_det, log_det_num, atol=ATOL
    ), f'Log determinant mismatch! Computed: {log_det}, Numerical: {log_det_num}'


# -----------------------
# Coupling Flow
# -----------------------
@pytest.mark.parametrize('n_blocks', [1, 3, 5])
def test_coupling_flow(batch, context, subnet_kwargs, transform_cfg, n_blocks):
    transform, rq_kwargs = transform_cfg
    x, mask = random_mask(batch)
    model = CouplingFlow(
        d_target=x.shape[-1],
        d_context=context.shape[-1],
        n_blocks=n_blocks,
        use_actnorm=True,
        use_permute=False,
        transform=transform,
        subnet_kwargs=subnet_kwargs,
        rq_kwargs=rq_kwargs,
    )
    model.eval()
    z, _, _ = model.forward(x, context=context, mask=mask)
    x_rec, _, _ = model.inverse(z, context, mask=mask)
    assert torch.isfinite(z).all(), 'anomaly in forward pass'
    assert torch.isfinite(x_rec).all(), 'anomaly in backward pass'
    # adaptive tails with random (untrained) weights compound across blocks;
    # tight tolerance only meaningful for zero_init or non-adaptive
    adaptive_tails = rq_kwargs.get('adaptive_tails', False)
    tight = not adaptive_tails or subnet_kwargs.get('zero_init', False)
    if tight:
        assert torch.allclose(x, x_rec, atol=ATOL), 'recovery failed'
    x = x.detach().clone().requires_grad_(True)
    z, log_det, _ = model.forward(x, context=context, mask=mask)
    assert torch.isfinite(log_det).all(), 'non-finite log-det'
    if tight:
        log_det_num = _numerical_logdet(z, x)
        assert torch.allclose(
            log_det, log_det_num, atol=ATOL
        ), f'Log determinant mismatch! Computed: {log_det}, Numerical: {log_det_num}'
    x_samp, log_prob = model.sample(10, context=context, mask=mask)
    assert torch.isfinite(x_samp).all(), 'anomaly in samples'
    assert torch.isfinite(log_prob).all(), 'anomaly in log_prob'


# -----------------------
# Spline stability: fixed vs adaptive domain, 6 blocks, no zero-init
# -----------------------
@pytest.mark.parametrize(
    'rq_kwargs',
    [
        {},
        {'adaptive_tails': True},
        {'adaptive_domain': True, 'adaptive_tails': True},
    ],
    ids=['fixed_domain', 'fixed_domain_adaptive_tails', 'adaptive_domain'],
)
def test_spline_stability(context, rq_kwargs):
    """
    6 blocks, no zero-init.
    Fixed-domain spline must meet tight ATOL (no compounding).
    Adaptive-domain spline with learned tails has weights that compound before
    training constrains them; we verify invertibility and finite log-det only.
    """
    torch.manual_seed(0)
    x_raw = torch.randn(B, 10, 3)
    x, mask = random_mask(x_raw)
    subnet_kwargs = {'net_type': 'mlp', 'zero_init': False}
    model = CouplingFlow(
        d_target=x.shape[-1],
        d_context=context.shape[-1],
        n_blocks=6,
        use_actnorm=True,
        use_permute=False,
        transform='spline',
        subnet_kwargs=subnet_kwargs,
        rq_kwargs=rq_kwargs,
    )
    model.eval()

    z, _, _ = model.forward(x, context=context, mask=mask)
    x_rec, _, _ = model.inverse(z, context=context, mask=mask)
    assert torch.isfinite(z).all(), 'anomaly in forward pass'
    assert torch.isfinite(x_rec).all(), 'anomaly in backward pass'

    x_g = x.detach().clone().requires_grad_(True)
    z_g, log_det, _ = model.forward(x_g, context=context, mask=mask)
    assert torch.isfinite(log_det).all(), 'non-finite log-det'

    if not rq_kwargs.get('adaptive_tails', False):
        # fixed domain, no adaptive tails: must be numerically tight
        err = (log_det - _numerical_logdet(z_g, x_g)).abs()
        assert torch.allclose(x, x_rec, atol=ATOL), 'recovery failed'
        assert err.max() < ATOL, f'log-det mismatch: mean={err.mean():.4f}, max={err.max():.4f}'
