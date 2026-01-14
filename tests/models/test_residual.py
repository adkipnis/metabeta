import pytest
import torch

from metabeta.models.feedforward.residual import (
    ResidualBlock,
    ResidualNet,
    FlowResidualNet,
)


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def batch():
    return torch.randn(4, 16)


@pytest.fixture
def context():
    return torch.randn(4, 8)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------
# ResidualBlock
# -----------------------

def test_residualblock_output_shape(batch):
    block = ResidualBlock(d_hidden=16)
    out = block(batch)
    assert out.shape == batch.shape


def test_residualblock_with_context_and_glu(batch, context):
    block = ResidualBlock(d_hidden=16, d_context=8, use_glu=True)
    out = block(batch, context)
    assert out.shape == batch.shape


def test_residualblock_rscale_positive(batch):
    block = ResidualBlock(d_hidden=16)
    out = block(batch)
    rscale = block.rscale
    assert torch.all(rscale > 0)
    assert torch.all(rscale <= 1)


def test_residualblock_backward(batch):
    block = ResidualBlock(d_hidden=16)
    out = block(batch)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in block.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


# -----------------------
# ResidualNet
# -----------------------

def test_residualnet_forward(batch):
    net = ResidualNet(d_input=16, d_hidden=32, d_output=8, depth=2)
    out = net(batch)
    assert out.shape == (batch.shape[0], 8)


def test_residualnet_forward_with_context(batch, context):
    net = ResidualNet(d_input=16, d_hidden=32, d_output=8, depth=2, d_context=8, use_glu=True)
    out = net(batch, context)
    assert out.shape == (batch.shape[0], 8)


def test_residualnet_backward(batch):
    net = ResidualNet(d_input=16, d_hidden=32, d_output=8, depth=2)
    out = net(batch)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_residualnet_forward_and_grads(batch):
    net = ResidualNet(d_input=16, d_hidden=32, d_output=8, depth=2)
    out = net(batch)
    
    # check output numerics
    assert torch.isfinite(out).all()
    
    # backward
    loss = out.mean()
    loss.backward()
    
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    # at least one grad exists
    assert any(g is not None for g in grads)
    
    # check all grads are finite (skip None)
    for g in grads:
        if g is not None:
            assert torch.isfinite(g).all()


# -----------------------
# FlowResidualNet
# -----------------------

def test_flowresidualnet_forward(batch):
    net = FlowResidualNet(d_input=16, d_hidden=32, d_output=16, depth=2)
    out = net(batch)
    assert out.shape == (batch.shape[0], 16)


def test_flowresidualnet_forward_with_context(batch, context):
    net = FlowResidualNet(d_input=16, d_hidden=32, d_output=16, depth=2, d_context=8)
    out = net(batch, context)
    assert out.shape == (batch.shape[0], 16)


def test_flowresidualnet_backward(batch):
    net = FlowResidualNet(d_input=16, d_hidden=32, d_output=16, depth=2)
    out = net(batch)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    
