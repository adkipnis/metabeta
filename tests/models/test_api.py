"""Public-`Api` tests on DataFrames: sample → posteriorSummary / rfxSummary / plotParameters.

Every test goes through `Api` on the tiny-Approximator joint checkpoint, never a private helper.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from metabeta.models.api import Api
from test_router import _write_tiny_approximator_checkpoint


@pytest.fixture(scope='module')
def api(tmp_path_factory: pytest.TempPathFactory) -> Api:
    path: Path = tmp_path_factory.mktemp('api') / 'joint.pt'
    torch.manual_seed(0)
    _write_tiny_approximator_checkpoint(
        path, max_d=4, max_q=2, likelihood_family=0, analytical_refinement='light'
    )
    return Api(path, warmup=False)


def _sleepstudy(m: int = 12, n_i: int = 10, seed: int = 0, **extra: np.ndarray) -> pd.DataFrame:
    """Reaction times over Days for m subjects, plus the columns given in `extra`."""
    rng = np.random.default_rng(seed)
    days = np.tile(np.arange(n_i, dtype=float), m)
    intercepts = np.repeat(rng.normal(0, 25, m), n_i)
    slopes = np.repeat(rng.normal(0, 6, m), n_i)
    reaction = 250 + 10 * days + intercepts + slopes * days + rng.normal(0, 25, m * n_i)
    subject = np.repeat([f's{i}' for i in range(m)], n_i)
    return pd.DataFrame({'Reaction': reaction, 'Days': days, 'Subject': subject, **extra})


def test_column_names_keep_the_dataframe_spelling(api: Api):
    rng = np.random.default_rng(1)
    treatment = np.where(rng.random(120) < 0.3, 'drug', 'ctrl')
    df = _sleepstudy(Treatment=treatment)

    result = api.sample(
        df,
        formula='Reaction ~ Days + Treatment + (1 + Days | Subject)',
        n_samples=8,
        fit_preprocessor=True,
    )

    assert result.param_names == {
        'ffx': ['Intercept', 'Days', 'Treatment_drug'],
        'sigma_rfx': ['Intercept', 'Days'],
    }
    summary = api.posteriorSummary(result)
    assert '| Days ' in summary and '| Treatment_drug ' in summary
    assert 'days' not in summary and 'treatment' not in summary
    rfx = api.rfxSummary(result)
    assert '\nDays:\n' in rfx and 'days' not in rfx
    grid = api.plotParameters(result)
    labels = [ax.get_ylabel() for ax in grid.axes[:, 0]]
    plt.close(grid.figure)
    assert 'Days' in labels and any('Treatment' in label for label in labels)
    assert not any('days' in label or 'treatment' in label for label in labels)


def test_formula_without_fixed_terms_fits_the_intercept_only(api: Api):
    df = _sleepstudy(Noise=np.random.default_rng(1).normal(size=120))

    result = api.sample(
        df, formula='Reaction ~ 1 + (1 | Subject)', n_samples=8, fit_preprocessor=True
    )

    assert result.param_names == {'ffx': ['Intercept'], 'sigma_rfx': ['Intercept']}


def test_no_formula_uses_every_other_column_as_a_fixed_effect(api: Api):
    df = _sleepstudy(Noise=np.random.default_rng(1).normal(size=120))

    result = api.sample(
        df.rename(columns={'Reaction': 'y'}),
        group_name='Subject',
        n_samples=8,
        fit_preprocessor=True,
    )

    assert result.param_names == {'ffx': ['Intercept', 'Days', 'Noise'], 'sigma_rfx': ['Intercept']}
