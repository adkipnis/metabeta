"""Tests for the Api's default priors and the trained-range check (#53)."""

import numpy as np
import pytest

from metabeta.utils.priors import (
    TRAINED_RANGE,
    apiDefaultPriors,
    bambiDefaultPriors,
    outsideTrainedRange,
)


def _batch(priors: list[dict[str, np.ndarray]], max_d: int, max_q: int) -> dict[str, np.ndarray]:
    """Batch-stage arrays of single-dataset priors, padded to max_d and max_q."""

    def padded(key: str, width: int) -> np.ndarray:
        return np.stack([np.pad(p[key], (0, width - len(p[key]))) for p in priors])

    hyper = {
        'nu_ffx': padded('nu_ffx', max_d),
        'tau_ffx': padded('tau_ffx', max_d),
        'tau_rfx': padded('tau_rfx', max_q),
        'mask_d': padded('tau_ffx', max_d) > 0,
        'mask_q': padded('tau_rfx', max_q) > 0,
        'eta_rfx': np.array([p['eta_rfx'] for p in priors]),
    }
    if 'tau_eps' in priors[0]:
        hyper['tau_eps'] = np.array([p['tau_eps'] for p in priors])
    return hyper


@pytest.mark.parametrize('family', [0, 1, 2])
def test_api_defaults_lie_inside_the_trained_range(family: int):
    priors = [apiDefaultPriors(d, q, family) for d, q in ((1, 1), (3, 2), (5, 3))]
    assert (outsideTrainedRange(_batch(priors, 6, 3), family) == '').all()


def test_bambi_scales_are_named_where_they_leave_the_range():
    priors = [bambiDefaultPriors(3, 2, 2), apiDefaultPriors(3, 2, 2)]
    outside = outsideTrainedRange(_batch(priors, 4, 2), 2)
    lo, hi = TRAINED_RANGE[2]['tau_rfx']
    assert outside[0] == f'tau_ffx 2.5 outside [0.0074, 1.42]; tau_rfx 2.5 outside [{lo:g}, {hi:g}]'
    assert outside[1] == ''


def test_padding_and_uncorrelated_rfx_are_ignored():
    prior = apiDefaultPriors(2, 1, 0)
    assert prior['eta_rfx'] == 0
    assert outsideTrainedRange(_batch([prior], 8, 3), 0)[0] == ''
    wide = prior | {'eta_rfx': np.array(4.0)}
    assert outsideTrainedRange(_batch([wide], 8, 3), 0)[0] == 'eta_rfx 4 outside [1, 2]'
