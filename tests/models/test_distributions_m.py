
import pytest
import torch

from metabeta.models.normalizingflows.distributions import StaticDist, TrainableDist, BaseDist


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def batch_shape():
    return (8, 3)  # (batch, d_data)


# -----------------------
# StaticDist Tests
# -----------------------

@pytest.mark.parametrize('family', ['normal', 'student'])
def test_staticdist_sample_shape(batch_shape, family):
    dist = StaticDist(d_data=batch_shape[1], family=family)
    x = dist.sample(batch_shape)
    assert x.shape == batch_shape
    assert isinstance(x, torch.Tensor)
    assert torch.isfinite(x).all()


@pytest.mark.parametrize('family', ['normal', 'student'])
def test_staticdist_logprob(batch_shape, family):
    dist = StaticDist(d_data=batch_shape[1], family=family)
    x = dist.sample(batch_shape)
    lp = dist.logProb(x)
    assert lp.shape == batch_shape
    assert torch.isfinite(lp).all()


# -----------------------
# TrainableDist Tests
# -----------------------

@pytest.mark.parametrize('family', ['normal', 'student'])
def test_traindist_sample_shape(batch_shape, family):
    dist = TrainableDist(d_data=batch_shape[1], family=family)
    x = dist.sample(batch_shape)
    assert x.shape == batch_shape
    assert torch.isfinite(x).all()


@pytest.mark.parametrize('family', ['normal', 'student'])
def test_traindist_logprob(batch_shape, family):
    dist = TrainableDist(d_data=batch_shape[1], family=family)
    x = dist.sample(batch_shape)
    lp = dist.logProb(x)
    assert lp.shape == batch_shape
    assert torch.isfinite(lp).all()


def test_traindist_invalid_sample_shape():
    dist = TrainableDist(d_data=3, family='normal')
    with pytest.raises(ValueError):
        dist.sample((4, 4))  # last dim != d_data


# -----------------------
# BaseDist Tests
# -----------------------

@pytest.mark.parametrize('trainable', [False, True])
@pytest.mark.parametrize('family', ['normal', 'student'])
def test_basedist_sample_and_logprob(batch_shape, trainable, family):
    dist = BaseDist(d_data=batch_shape[1], family=family, trainable=trainable)
    x = dist.sample(batch_shape)
    lp = dist.logProb(x)
    assert x.shape == batch_shape
    assert lp.shape == batch_shape
    assert torch.isfinite(x).all()
    assert torch.isfinite(lp).all()


@pytest.mark.parametrize('trainable', [False, True])
@pytest.mark.parametrize('family', ['normal', 'student'])
def test_basedist_repr(trainable, family):
    dist = BaseDist(d_data=3, family=family, trainable=trainable)
    s = repr(dist)
    assert isinstance(s, str)
    assert family in s.lower()
