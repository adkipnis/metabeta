import torch
from metabeta.models.normalizingflows import LU, Permute
from metabeta.models.normalizingflows.coupling import (
    Coupling, DualCoupling, CouplingFlow
)

ATOL = 1e-5
NET_KWARGS = {
    'net_type': 'mlp',
    'd_ff': 128,
    'depth': 3,
    'activation': 'ReLU',
    'zero_init': False, # if True, the initial flows are identity maps
}

def test_lu():
    x = torch.randn((8, 10, 3))
    x[0, 0, -1] = 0.0
    mask = (x != 0.0).float()

    model = LU(3, identity_init=False)
    model.eval()
    z, log_det, _ = model.forward(x, mask=mask)
    z_, _, _ = model.inverse(z, mask=mask)
    assert torch.allclose(x, z_, atol=ATOL), 'LU is not invertible'

    x.requires_grad_(True)
    z_numerical, _, _ = model.forward(x, mask=mask)
    jacobian = []
    for i in range(z_numerical.shape[-1]):
        grad_z_i = torch.autograd.grad(z_numerical[..., i].sum(), x, retain_graph=True)[0]
        jacobian.append(grad_z_i)
    jacobian = torch.stack(jacobian, dim=-1)
    numerical_log_det = torch.log(torch.abs(torch.det(jacobian)))
    assert torch.allclose(log_det, numerical_log_det, atol=ATOL), (
        f'Log determinant mismatch! Computed: {log_det}, Numerical: {numerical_log_det}'
    )
    print('LU Transform passed all tests!')


def test_single_coupling():
    inputs = torch.randn((8, 3))
    x1, x2 = inputs.chunk(2, dim=-1)
    split_dims = (x1.shape[-1], x2.shape[-1])
    model1 = Coupling(split_dims, net_kwargs=NET_KWARGS)
    model2 = Coupling((split_dims[1], split_dims[0]), net_kwargs=NET_KWARGS)

    model1.eval()
    model2.eval()
    (z1, z2), _ = model1.forward(x1, x2)
    (z1, z2), _ = model1.inverse(z1, z2)
    assert (
        torch.allclose(x1, z1, atol=ATOL) and
        torch.allclose(x2, z2, atol=ATOL)
    ), 'model1 is not invertible'

    (z2, z1), _ = model2.forward(x2, x1)
    (z2, z1), _ = model2.inverse(z2, z1)
    assert (
        torch.allclose(x1, z1, atol=ATOL) and
        torch.allclose(x2, z2, atol=ATOL)
    ), 'model2 is not invertible'

    (z1, z2), _ = model1(x1, x2)
    (z2, z1), _ = model2(z2, z1)
    (z2, z1), _ = model2.inverse(z2, z1)
    (z1, z2), _ = model1.inverse(z1, z2)
    assert (
        torch.allclose(x1, z1, atol=ATOL) and
        torch.allclose(x2, z2, atol=ATOL)
    ), 'serial model is not invertible'

    context  = torch.randn((8, 5))
    model3 = Coupling(split_dims, d_context=5, net_kwargs=NET_KWARGS)
    model3.eval()
    (z1, z2), _ = model3.forward(x1, x2, context)
    (z1, z2), _ = model3.inverse(z1, z2, context)
    assert (
        torch.allclose(x1, z1, atol=ATOL) and
        torch.allclose(x2, z2, atol=ATOL)
    ), 'conditional model is not invertible'

    print('Single Coupling passed all tests!')


def test_dual_coupling():
    x = torch.randn((8, 3))
    context = torch.randn((8, 5))
    model = DualCoupling(3, d_context=5, net_kwargs=NET_KWARGS)
    model.eval()
    z, log_det, _ = model.forward(x, context)
    z, _, _ = model.inverse(z, context)
    assert torch.allclose(x, z, atol=ATOL), 'model is not invertible'

    x.requires_grad_(True)
    z_numerical, _, _ = model.forward(x, context)
    jacobian = []
    for i in range(z_numerical.shape[-1]):
        grad_z_i = torch.autograd.grad(z_numerical[:, i].sum(), x, retain_graph=True)[0]
        jacobian.append(grad_z_i)
    jacobian = torch.stack(jacobian, dim=-1)
    numerical_log_det = torch.log(torch.abs(torch.det(jacobian)))
    assert torch.allclose(log_det, numerical_log_det, atol=ATOL), (
        f'Log determinant mismatch! Computed: {log_det}, Numerical: {numerical_log_det}'
    )

    print('Dual Coupling passed all tests!')


def test_coupling_flow():
    x = torch.randn((8, 3))
    context = torch.randn((8, 5))
    model = CouplingFlow(3, d_context=5, n_blocks=3,
                         use_actnorm=True, net_kwargs=NET_KWARGS)
    model.eval()
    z, log_det, _ = model.forward(x, context)
    z, _, _ = model.inverse(z, context)
    assert torch.allclose(x, z, atol=ATOL), 'model is not invertible'

    x.requires_grad_(True)
    z_numerical, _, _ = model.forward(x, context)
    jacobian = []
    for i in range(z_numerical.shape[-1]):
        grad_z_i = torch.autograd.grad(z_numerical[:, i].sum(), x, retain_graph=True)[0]
        jacobian.append(grad_z_i)
    jacobian = torch.stack(jacobian, dim=-1)
    numerical_log_det = torch.log(torch.abs(torch.det(jacobian)))
    assert torch.allclose(log_det, numerical_log_det, atol=1e-5), (
        f'Log determinant mismatch! Computed: {log_det}, Numerical: {numerical_log_det}'
    )

    x_, log_q = model.sample(100, context)
    assert x_.shape == (8, 100, 3), 'sample shape is off'

    print('Coupling Flow passed all tests!')


def test_masking():
    x = torch.randn((8, 3))
    x[0, -1] = 0.0
    mask = (x != 0.0).float()

    model = Permute(3)
    _, _, mask_ = model(x, mask=mask)
    _, _, mask_ = model.inverse(x, mask=mask_)
    assert mask_ is not None, 'mask should be a tensor'
    assert torch.allclose(mask, mask_, atol=ATOL), 'mask is not recovered properly'

    model = LU(3, identity_init=False)
    z, log_det, mask_ = model(x, mask=mask)
    assert z[0, -1] == 0., 'mask not properly applied to z'
    z, log_det_, mask_ = model.inverse(z, mask=mask_)
    assert torch.allclose(x, z, atol=ATOL), 'x is not recovered properly'

    model = Coupling((2, 1), net_kwargs=NET_KWARGS)
    model.eval()
    x1, x2 = x.chunk(2, dim=-1)
    _, mask2 = mask.chunk(2, dim=-1)
    (z1, z2), log_det = model(x1, x2)
    if not NET_KWARGS['zero_init']:
        assert z2[0, 0] != 0.0, 'z should not be 0'
        assert log_det[0] != 0.0, 'log_det should not be zero'

    (z1, z2), log_det = model(x1, x2, mask2=mask2)
    assert x2[0, 0] == z2[0, 0] == 0.0, 'mask not properly applied to z'
    assert log_det[0] == 0.0, 'log_det not properly masked'

    (z1, z2), log_det = model.inverse(z1, z2, mask2=mask2)
    assert x2[0, 0] == z2[0, 0] == 0.0, 'mask not properly applied to z'
    assert log_det[0] == 0.0, 'log_det not properly masked'

    model = DualCoupling(3, net_kwargs=NET_KWARGS)
    model.eval()
    z, log_det, _ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'
    if not NET_KWARGS['zero_init']:
        assert log_det[0] != 0.0, 'log_det should not be zero'
        z, log_det_, _ = model(x)
        assert log_det[0] != log_det_[0], 'log_det should differ at first index'

    model = CouplingFlow(
        3,
        n_blocks=1,
        use_actnorm=True,
        use_permute=False,
        net_kwargs=NET_KWARGS,
    )
    # model = model.to('mps')
    # x = x.to('mps')
    # mask = mask.to('mps')
    model.eval()
    z, log_det, mask_ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'
    z, log_det_, mask_ = model.inverse(z, mask=mask_)
    assert mask_ is not None, 'mask_ should be a tensor'
    assert torch.allclose(mask, mask_, atol=ATOL), 'mask is not recovered properly'
    assert torch.allclose(x, z, atol=ATOL), 'x is not recovered properly'

    x_, log_q = model.sample(100, mask=mask)
    assert (x_[0, :, -1] == 0.0).all(), 'mask not properly applied during sampling'

    print('Masked Coupling Flow passed all tests!')


def test_rq():
    NET_KWARGS['zero_init'] = True
    x = torch.randn((8, 3)) * 3
    x[0, -1] = 0.0
    mask = (x != 0.0).float()

    model = Coupling((2, 1), net_kwargs=NET_KWARGS, transform='spline')
    model.eval()
    x1, x2 = x.chunk(2, dim=-1)
    _, mask2 = mask.chunk(2, dim=-1)
    (z1, z2), log_det = model(x1, x2)
    if not NET_KWARGS['zero_init']:
        assert z2[0, 0] != 0.0, 'z should not be 0'
        assert log_det[0] != 0.0, 'log_det should not be zero'

    (z1, z2), log_det = model(x1, x2, mask2=mask2)
    assert x2[0, 0] == z2[0, 0] == 0.0, 'mask not properly applied to z'
    assert log_det[0] == 0.0, 'log_det not properly masked'

    (z1, z2), log_det = model.inverse(z1, z2, mask2=mask2)
    assert x2[0, 0] == z2[0, 0] == 0.0, 'mask not properly applied to z'
    assert log_det[0] == 0.0, 'log_det not properly masked'

    model = DualCoupling(3, net_kwargs=NET_KWARGS, transform='spline')
    model.eval()
    z, log_det, _ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'
    if not NET_KWARGS['zero_init']:
        assert log_det[0] != 0.0, 'log_det should not be zero'
        z, log_det_, _ = model(x)
        assert log_det[0] != log_det_[0], 'log_det should differ at first index'

    model = CouplingFlow(
        3,
        n_blocks=1,
        use_actnorm=True,
        use_permute=False,
        net_kwargs=NET_KWARGS,
        transform='spline',
    )
    # model = model.to('mps')
    # x = x.to('mps')
    # mask = mask.to('mps')
    model.eval()
    z, log_det, mask_ = model(x, mask=mask)
    assert z[0, -1] == 0.0, 'mask not properly applied to z'
    z, log_det_, mask_ = model.inverse(z, mask=mask_)
    assert mask_ is not None, 'mask_ should be a tensor'
    assert torch.allclose(mask, mask_, atol=ATOL), 'mask is not recovered properly'
    assert torch.allclose(x, z, atol=ATOL), 'x is not recovered properly'

    x_, log_q = model.sample(100, mask=mask)
    assert (x_[0, :, -1] == 0.0).all(), 'mask not properly applied during sampling'

    print('Masked Neural Spline Flow passed all tests!')



if __name__ == '__main__':
    torch.manual_seed(0)
    test_lu()
    test_single_coupling()
    test_dual_coupling()
    test_coupling_flow()
    test_masking()
    test_rq()

