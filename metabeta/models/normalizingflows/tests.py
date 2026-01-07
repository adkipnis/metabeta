import torch
from metabeta.models.normalizingflows import LU
from metabeta.models.normalizingflows.coupling import (
    Coupling, DualCoupling
)

ATOL = 1e-5
NET_KWARGS = {
    'net_type': 'mlp',
    'd_ff': 128,
    'depth': 3,
    'activation': 'ReLU',
}

def test_lu():
    x = torch.randn((8, 3))
    x[0, -1] = 0.0
    mask = (x != 0.0).float()

    model = LU(3, identity_init=False)
    model.eval()
    z, log_det, _ = model.forward(x, mask=mask)
    z_, _, _ = model.inverse(z, mask=mask)
    assert torch.allclose(x, z_, atol=ATOL), "LU is not invertible"

    x.requires_grad_(True)
    z_numerical, _, _ = model.forward(x, mask=mask)
    jacobian = []
    for i in range(z_numerical.shape[-1]):
        grad_z_i = torch.autograd.grad(z_numerical[:, i].sum(), x, retain_graph=True)[0]
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

    condition = torch.randn((8, 5))
    model3 = Coupling(split_dims, d_context=5, net_kwargs=NET_KWARGS)
    model3.eval()
    (z1, z2), _ = model3.forward(x1, x2, condition)
    (z1, z2), _ = model3.inverse(z1, z2, condition)
    assert (
        torch.allclose(x1, z1, atol=ATOL) and
        torch.allclose(x2, z2, atol=ATOL)
    ), 'conditional model is not invertible'

    print('Single Coupling passed all tests!')


def test_dual_coupling():
    x = torch.randn((8, 3))
    condition = torch.randn((8, 5))
    model = DualCoupling(3, d_context=5, net_kwargs=NET_KWARGS)
    model.eval()
    z, log_det, _ = model.forward(x, condition)
    z, _, _ = model.inverse(z, condition)
    assert torch.allclose(x, z, atol=1e-5), 'model is not invertible'

    x.requires_grad_(True)
    z_numerical, _, _ = model.forward(x, condition)
    jacobian = []
    for i in range(z_numerical.shape[-1]):
        grad_z_i = torch.autograd.grad(z_numerical[:, i].sum(), x, retain_graph=True)[0]
        jacobian.append(grad_z_i)
    jacobian = torch.stack(jacobian, dim=-1)
    numerical_log_det = torch.log(torch.abs(torch.det(jacobian)))
    assert torch.allclose(log_det, numerical_log_det, atol=1e-5), (
        f'Log determinant mismatch! Computed: {log_det}, Numerical: {numerical_log_det}'
    )

    print('Dual Coupling passed all tests!')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_lu()
    test_single_coupling()
    test_dual_coupling()

