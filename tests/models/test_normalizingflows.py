from __future__ import annotations

import pytest
import torch

from metabeta.models.normalizingflows import LU, Permute
from metabeta.models.normalizingflows.coupling import Coupling, DualCoupling, CouplingFlow

ATOL = 1e-5


@pytest.fixture(autouse=True)
def _torch_seed() -> None:
    torch.manual_seed(1)
    torch.set_printoptions(precision=5)


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


def _numerical_logdet(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample log|det J| by explicitly building the Jacobian.

    Note: only intended for small tensors in tests.
    """
    assert x.requires_grad
    jac_rows = []
    for i in range(z.shape[-1]):
        grad_z_i = torch.autograd.grad(z[..., i].sum(), x, retain_graph=True)[0]
        jac_rows.append(grad_z_i)
    jac = torch.stack(jac_rows, dim=-1)
    return torch.log(torch.abs(torch.det(jac)))


def test_lu_invertible_and_logdet_matches_jacobian():
    x = torch.randn((8, 10, 3))
    x[0, 0, -1] = 0.0
    mask = (x != 0.0).float()

    model = LU(3, identity_init=False)
    model.eval()

    z, log_det, _ = model.forward(x, mask=mask)
    x_rec, _, _ = model.inverse(z, mask=mask)
    assert torch.allclose(x, x_rec, atol=ATOL), 'LU is not invertible'

    x = x.detach().clone().requires_grad_(True)
    z_num, log_det_num, _ = model.forward(x, mask=mask)
    num_log_det = _numerical_logdet(z_num, x)
    assert torch.allclose(log_det_num, num_log_det, atol=ATOL), (
        f'Log determinant mismatch! Computed: {log_det_num}, Numerical: {num_log_det}'
    )


@pytest.mark.parametrize('swap', [False, True])
def test_coupling_invertible(swap: bool, subnet_kwargs: dict[str, object]):
    inputs = torch.randn((8, 3), dtype=torch.float64)
    x1, x2 = inputs.chunk(2, dim=-1)
    split_dims = (x1.shape[-1], x2.shape[-1])

    dims = (split_dims[1], split_dims[0]) if swap else split_dims
    model = Coupling(dims, subnet_kwargs=subnet_kwargs).double()
    model.eval()

    if swap:
        (z2, z1), _ = model.forward(x2, x1)
        (x2_rec, x1_rec), _ = model.inverse(z2, z1)
        assert torch.allclose(x1, x1_rec, atol=ATOL), 'coupling is not invertible (x1)'
        assert torch.allclose(x2, x2_rec, atol=ATOL), 'coupling is not invertible (x2)'
    else:
        (z1, z2), _ = model.forward(x1, x2)
        (x1_rec, x2_rec), _ = model.inverse(z1, z2)
        assert torch.allclose(x1, x1_rec, atol=ATOL), 'coupling is not invertible (x1)'
        assert torch.allclose(x2, x2_rec, atol=ATOL), 'coupling is not invertible (x2)'


def test_serial_coupling_invertible_and_conditional_invertible(subnet_kwargs: dict[str, object]):
    inputs = torch.randn((8, 3), dtype=torch.float64)
    x1, x2 = inputs.chunk(2, dim=-1)
    split_dims = (x1.shape[-1], x2.shape[-1])

    model1 = Coupling(split_dims, subnet_kwargs=subnet_kwargs).double()
    model2 = Coupling((split_dims[1], split_dims[0]), subnet_kwargs=subnet_kwargs).double()
    model1.eval()
    model2.eval()

    (z1_, z2_), _ = model1(x1, x2)
    (z2, z1), _ = model2(z2_, z1_)
    (z2, z1), _ = model2.inverse(z2, z1)
    (x1_rec, x2_rec), _ = model1.inverse(z1, z2)

    assert torch.allclose(x1, x1_rec, atol=ATOL), 'serial coupling is not invertible (x1)'
    assert torch.allclose(x2, x2_rec, atol=ATOL), 'serial coupling is not invertible (x2)'

    context = torch.randn((8, 5), dtype=torch.float64)
    model3 = Coupling(split_dims, d_context=5, subnet_kwargs=subnet_kwargs).double()
    model3.eval()

    (z1, z2), _ = model3.forward(x1, x2, context)
    (x1_rec, x2_rec), _ = model3.inverse(z1, z2, context)

    assert torch.allclose(x1, x1_rec, atol=ATOL), 'conditional coupling is not invertible (x1)'
    assert torch.allclose(x2, x2_rec, atol=ATOL), 'conditional coupling is not invertible (x2)'


def test_dual_coupling_invertible_and_logdet_matches_jacobian(subnet_kwargs: dict[str, object]):
    x = torch.randn((8, 3), dtype=torch.float64)
    context = torch.randn((8, 5), dtype=torch.float64)

    model = DualCoupling(3, d_context=5, subnet_kwargs=subnet_kwargs).double()
    model.eval()

    z, log_det, _ = model.forward(x, context)
    x_rec, _, _ = model.inverse(z, context)
    assert torch.allclose(x, x_rec, atol=ATOL), 'dual coupling is not invertible'

    x = x.detach().clone().requires_grad_(True)
    z_num, log_det_num, _ = model.forward(x, context)
    num_log_det = _numerical_logdet(z_num, x)
    assert torch.allclose(log_det_num, num_log_det, atol=ATOL), (
        f'Log determinant mismatch! Computed: {log_det_num}, Numerical: {num_log_det}'
    )


def test_coupling_flow_invertible_logdet_and_sample_shape(subnet_kwargs: dict[str, object]):
    x = torch.randn((8, 3), dtype=torch.float64)
    context = torch.randn((8, 5), dtype=torch.float64)

    model = CouplingFlow(
        3,
        d_context=5,
        n_blocks=8,
        use_actnorm=False,
        use_permute=False,
        transform='spline',
        subnet_kwargs=subnet_kwargs,        
    ).double()
    model.eval()

    z, log_det, _ = model.forward(x, context)
    x_rec, _, _ = model.inverse(z, context)
    assert torch.allclose(x, x_rec, atol=ATOL), 'coupling flow is not invertible'

    x = x.detach().clone().requires_grad_(True)
    z_num, log_det_num, _ = model.forward(x, context)
    num_log_det = _numerical_logdet(z_num, x)
    assert torch.allclose(log_det_num, num_log_det, atol=ATOL), (
        f'Log determinant mismatch! Computed: {log_det_num}, Numerical: {num_log_det}'
    )

    x_samp, log_q = model.sample(100, context)
    assert x_samp.shape == (8, 100, 3), 'sample shape is off'
    assert log_q.shape[0] == 8, 'log_q batch dim is off'


def test_masking_invertibility_and_mask_roundtrip(subnet_kwargs: dict[str, object]):
    x = torch.randn((16, 3))
    x[0, -1] = 0.0
    mask = (x != 0.0).float()

    # permute roundtrips mask
    model = Permute(3)
    _, _, mask_ = model(x, mask=mask)
    _, _, mask_ = model.inverse(x, mask=mask_)
    assert mask_ is not None, 'mask should be a tensor'
    assert torch.allclose(mask, mask_, atol=ATOL), 'mask is not recovered properly'

    # LU respects mask and is invertible
    model = LU(3, identity_init=False)
    z, _, mask_ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'
    x_rec, _, _ = model.inverse(z, mask=mask_)
    assert torch.allclose(x, x_rec, atol=ATOL), 'x is not recovered properly'

    # Coupling respects mask2
    model = Coupling((2, 1), subnet_kwargs=subnet_kwargs)
    model.eval()

    x1, x2 = x.chunk(2, dim=-1)
    _, mask2 = mask.chunk(2, dim=-1)

    (z1, z2), log_det = model(x1, x2)
    if not bool(subnet_kwargs['zero_init']):
        assert z2[0, 0] != 0.0, 'z should not be 0'
        assert log_det[0] != 0.0, 'log_det should not be zero'

    (z1m, z2m), log_det_m = model(x1, x2, mask2=mask2)
    assert x2[0, 0] == z2m[0, 0] == 0.0, 'mask not properly applied to z'
    assert log_det_m[0] == 0.0, 'log_det not properly masked'

    (x1_rec, x2_rec), log_det_inv = model.inverse(z1m, z2m, mask2=mask2)
    assert x2[0, 0] == x2_rec[0, 0] == 0.0, 'mask not properly applied in inverse'
    assert log_det_inv[0] == 0.0, 'inverse log_det not properly masked'

    # DualCoupling respects mask
    model = DualCoupling(3, subnet_kwargs=subnet_kwargs)
    model.eval()

    z, log_det, _ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'
    if not bool(subnet_kwargs['zero_init']):
        assert log_det[0] != 0.0, 'log_det should not be zero'
        _, log_det_unmasked, _ = model(x)
        assert log_det[0] != log_det_unmasked[0], 'masked/unmasked log_det should differ at first index'

    # CouplingFlow respects mask and sampling respects mask
    model = CouplingFlow(
        3,
        n_blocks=3,
        use_actnorm=True,
        use_permute=False,
        subnet_kwargs=subnet_kwargs,
        transform='affine',
    )
    model.eval()

    z, _, mask_ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'
    x_rec, _, mask_rec = model.inverse(z, mask=mask_)
    assert mask_rec is not None, 'mask_ should be a tensor'
    assert torch.allclose(mask, mask_rec, atol=ATOL), 'mask is not recovered properly'
    # print(f'max dif: {(x-x_rec).abs().max().item():.6f}')
    assert torch.allclose(x, x_rec, atol=5e-5), 'x is not recovered properly'
    x_samp, _ = model.sample(100, mask=mask)
    assert (x_samp[0, :, -1] == 0.0).all(), 'mask not properly applied during sampling'


def test_spline_transform_masking(subnet_kwargs: dict[str, object]):
    dtype = torch.float32
    x = torch.randn((16, 3)).to(dtype)
    x[0, -1] = 0.0
    mask = (x != 0.0).to(dtype)

    # Coupling spline respects mask2
    model = Coupling((2, 1), subnet_kwargs=subnet_kwargs, transform='spline').to(dtype)
    model.eval()

    x1, x2 = x.chunk(2, dim=-1)
    _, mask2 = mask.chunk(2, dim=-1)

    (z1m, z2m), log_det_m = model(x1, x2, mask2=mask2)
    assert x2[0, 0] == z2m[0, 0] == 0.0, 'mask not properly applied to z'
    assert log_det_m[0] == 0.0, 'log_det not properly masked'

    (x1_rec, x2_rec), log_det_inv = model.inverse(z1m, z2m, mask2=mask2)
    assert x2[0, 0] == x2_rec[0, 0] == 0.0, 'mask not properly applied in inverse'
    assert log_det_inv[0] == 0.0, 'inverse log_det not properly masked'

    # DualCoupling spline is invertible and respects mask
    model = DualCoupling(3, subnet_kwargs=subnet_kwargs, transform='spline').to(dtype)
    model.eval()

    z, _, _ = model.forward(x, mask=mask)
    x_rec, _, _ = model.inverse(z, mask=mask)
    assert torch.allclose(x, x_rec, atol=ATOL), 'dual coupling (spline) is not invertible'
    assert z[0, -1] == 0.0, 'mask not properly applied to z'

    # CouplingFlow spline respects mask and sampling respects mask
    model = CouplingFlow(
        3,
        n_blocks=4,
        use_actnorm=True,
        use_permute=False,
        subnet_kwargs=subnet_kwargs,
        transform='spline',
    ).to(dtype)
    model.eval()

    z, _, mask_ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'

    x_rec, _, mask_rec = model.inverse(z, mask=mask_)
    assert mask_rec is not None, 'mask_ should be a tensor'
    assert torch.allclose(mask, mask_rec, atol=ATOL), 'mask is not recovered properly'
    assert torch.allclose(x, x_rec, atol=5e-5), f'x is not recovered properly, max delta: {(x-x_rec).abs().max().item():.6f}'

    x_samp, _ = model.sample(100, mask=mask)
    assert (x_samp[0, :, -1] == 0.0).all(), 'mask not properly applied during sampling'

# torch.manual_seed(1)
# subnet_kwargs = {
#     'net_type': 'mlp',
#     'zero_init': False,
# }
# test_coupling_flow_invertible_logdet_and_sample_shape(subnet_kwargs)
# test_masking_invertibility_and_mask_roundtrip(subnet_kwargs)
# test_spline_transform_masking(subnet_kwargs)