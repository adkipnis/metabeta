
import pytest
import torch

from metabeta.models.transformers.attention import MHA, MAB, ISAB


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def batch():
    return torch.randn(8, 10, 16)  # (batch, seq_len, d_model)


@pytest.fixture
def mask():
    m = torch.ones((8, 10), dtype=torch.bool)
    return m


@pytest.fixture
def context(batch):
    # For testing optional z input in MHA/MAB
    return batch.clone()


# -----------------------
# MHA Tests
# -----------------------

def test_mha_forward_shape(batch):
    model = MHA(d_model=16)
    out = model(batch)
    assert out.shape == batch.shape
    assert torch.isfinite(out).all()


def test_mha_forward_with_key_value(batch, context):
    model = MHA(d_model=16)
    out = model(batch, context, context)
    assert out.shape == batch.shape
    assert torch.isfinite(out).all()


def test_mha_forward_with_mask(batch, mask):
    model = MHA(d_model=16)
    out1 = model(batch, mask=mask)
    mask_clone = mask.clone()
    mask_clone[..., 0] = False  # mask first element
    out2 = model(batch, mask=mask_clone)
    # outputs where mask is True should be similar if inputs unchanged
    assert torch.isfinite(out1).all()
    assert torch.isfinite(out2).all()


def test_mha_backward(batch):
    model = MHA(d_model=16)
    out = model(batch)
    loss = out.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    for g in grads:
        if g is not None:
            assert torch.isfinite(g).all()


def test_mha_mask_behavior(batch):
    model = MHA(d_model=16)
    mask = torch.ones((batch.shape[0], batch.shape[1]), dtype=torch.bool)
    out_full = model(batch, mask=mask)
    mask[..., 0] = False
    batch_mod = batch.clone()
    batch_mod[..., 0] = -99.  # entry should not affect output
    out_masked = model(batch_mod, mask=mask)
    # masked entries should not affect attended outputs
    assert torch.allclose(out_full[mask], out_masked[mask], atol=1e-5)


# -----------------------
# MAB Tests
# -----------------------

def test_mab_forward(batch, context):
    model = MAB(d_model=16, d_ff=64, pre_norm=False)
    out = model(batch)
    assert out.shape == batch.shape
    out_with_context = model(batch, z=context)
    assert out_with_context.shape == batch.shape
    assert torch.isfinite(out_with_context).all()


def test_mab_prenorm(batch, context):
    model = MAB(d_model=16, d_ff=64, pre_norm=True)
    out = model(batch)
    assert out.shape == batch.shape
    out_with_context = model(batch, z=context)
    assert out_with_context.shape == batch.shape
    assert torch.isfinite(out_with_context).all()


def test_mab_backward(batch):
    model = MAB(d_model=16, d_ff=64)
    out = model(batch)
    loss = out.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    for g in grads:
        if g is not None:
            assert torch.isfinite(g).all()


# -----------------------
# ISAB Tests
# -----------------------

def test_isab_forward_shape(batch, mask):
    model = ISAB(d_model=16, d_ff=64, n_inducing=32)
    out = model(batch)
    assert out.shape == batch.shape
    out_masked = model(batch, mask=mask)
    assert out_masked.shape == batch.shape
    assert torch.isfinite(out_masked).all()


def test_isab_backward(batch):
    model = ISAB(d_model=16, d_ff=64, n_inducing=32)
    out = model(batch)
    loss = out.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    for g in grads:
        if g is not None:
            assert torch.isfinite(g).all()

