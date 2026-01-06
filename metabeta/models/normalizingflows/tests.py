import torch
from metabeta.models.normalizingflows.coupling import (
    Coupling,
)

ATOL = 1e-5
NET_KWARGS = {
    'net_type': 'residual',
    'd_ff': 128,
    'depth': 3,
    'activation': 'ReLU',
}

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

    print("Single Coupling passed all tests!")


if __name__ == '__main__':
    torch.manual_seed(0)
    test_single_coupling()

