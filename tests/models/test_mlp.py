import pytest
import torch

from metabeta.models.feedforward.mlp import (
    Feedforward,
    MLP,
    TransformerFFN,
    FlowMLP,
    )

# -----------------------
# helpers / fixtures
# -----------------------

@pytest.fixture
def batch():
    return torch.randn(4, 16)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------
# Feedforward
# -----------------------

def test_feedforward_output_shape(batch):
    layer = Feedforward(
        d_input=16,
        d_output=32,
        layer_norm=True,
        activation='ReLU',
        dropout=0.0,
    )
    out = layer(batch)
    assert out.shape == (4, 32)


def test_feedforward_residual_only_when_dims_match():
    ff_residual = Feedforward(
        d_input=8,
        d_output=8,
        residual=True,
    )
    assert ff_residual.residual is True

    ff_no_residual = Feedforward(
        d_input=8,
        d_output=16,
        residual=True,
    )
    assert ff_no_residual.residual is False


def test_feedforward_residual_forward(batch):
    ff = Feedforward(
        d_input=16,
        d_output=16,
        residual=True,
    )
    out = ff(batch)
    assert out.shape == batch.shape


# -----------------------
# MLP
# -----------------------

def test_mlp_basic_forward(batch):
    mlp = MLP(
        d_input=16,
        d_hidden=[32, 32],
        d_output=8,
        dropout=0.0,
    )
    out = mlp(batch)
    assert out.shape == (4, 8)


def test_mlp_shortcut_adds_projection(batch):
    mlp = MLP(
        d_input=16,
        d_hidden=32,
        d_output=16,
        shortcut=True,
        dropout=0.0,
    )
    out = mlp(batch)
    assert out.shape == batch.shape


def test_mlp_without_shortcut(batch):
    mlp = MLP(
        d_input=16,
        d_hidden=32,
        d_output=8,
        shortcut=False,
        dropout=0.0,
    )
    out = mlp(batch)
    assert out.shape == (4, 8)


def test_mlp_backward_pass(batch):
    mlp = MLP(
        d_input=16,
        d_hidden=32,
        d_output=8,
    )
    out = mlp(batch)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in mlp.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_mlp_forward_and_grads(batch):
    mlp = MLP(
        d_input=16,
        d_hidden=32,
        d_output=8,
    )
    out = mlp(batch)
    
    # check output numerics
    assert torch.isfinite(out).all()
    
    # backward
    loss = out.mean()
    loss.backward()
    
    grads = [p.grad for p in mlp.parameters() if p.requires_grad]
    # at least one grad exists
    assert any(g is not None for g in grads)
    
    # check all grads are finite (skip None)
    for g in grads:
        if g is not None:
            assert torch.isfinite(g).all()


# -----------------------
# TransformerFFN
# -----------------------

def test_transformer_ffn_forward():
    x = torch.randn(2, 10)
    ffn = TransformerFFN(
        d_input=10,
        d_hidden=20,
        d_output=10,
    )
    out = ffn(x)
    assert out.shape == x.shape


def test_transformer_ffn_invalid_activation():
    with pytest.raises(AssertionError):
        TransformerFFN(
            d_input=8,
            d_hidden=16,
            d_output=8,
            activation='ReLU',  # invalid for TransformerFFN
        )


# -----------------------
# FlowMLP
# -----------------------

def test_flowmlp_without_context():
    x = torch.randn(3, 6)
    net = FlowMLP(
        d_input=6,
        d_hidden=12,
        d_output=6,
    )
    out = net(x)
    assert out.shape == x.shape


def test_flowmlp_with_context():
    x = torch.randn(3, 6)
    context = torch.randn(3, 4)

    net = FlowMLP(
        d_input=10,  # 6 + 4
        d_hidden=12,
        d_output=6,
    )
    out = net(x, context)
    assert out.shape == x.shape


def test_flowmlp_backward_pass():
    x = torch.randn(5, 4)
    net = FlowMLP(
        d_input=4,
        d_hidden=[8, 8],
        d_output=4,
    )
    out = net(x)
    loss = out.pow(2).mean()
    loss.backward()

    grads = [p.grad for p in net.parameters() if p.requires_grad and p.numel() > 0]
    assert all(g is not None for g in grads)
