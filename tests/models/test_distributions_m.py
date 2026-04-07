import pytest
import torch

from metabeta.models.normalizingflows.distributions import BaseDist


@pytest.fixture
def batch_shape():
    return (8, 3)  # (batch, d_data)


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
