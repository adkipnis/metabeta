"""Public-`Api` tests on DataFrames: sample → posteriorSummary / rfxSummary / plotParameters.

Every test goes through `Api` on the tiny-Approximator joint checkpoint, never a private helper.
"""

import logging
from pathlib import Path
import warnings

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


def test_preprocessing_ignores_a_collinear_column_the_formula_does_not_name(
    api: Api, caplog: pytest.LogCaptureFixture
):
    df = _sleepstudy()
    days2 = 2 * df['Days'] + np.random.default_rng(1).normal(0, 0.01, len(df))
    formula = 'Reaction ~ Days + (1 + Days | Subject)'

    torch.manual_seed(0)
    reference = api.sample(df, formula=formula, n_samples=8, fit_preprocessor=True)
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.INFO):
        warnings.simplefilter('always')
        result = api.sample(
            df.assign(Days2=days2), formula=formula, n_samples=8, fit_preprocessor=True
        )

    assert result.param_names == reference.param_names
    for key in ('ffx', 'sigma_rfx', 'rfx'):
        assert torch.equal(getattr(result.proposal, key), getattr(reference.proposal, key))
    messages = [str(w.message) for w in caught] + [r.getMessage() for r in caplog.records]
    assert not any('days2' in message.lower() for message in messages)


@pytest.mark.parametrize(
    ('formula', 'term'),
    [
        ('Reaction ~ Noise + (1 + Days | Subject)', 'Days'),
        ('Reaction ~ 0 + Days + (1 | Subject)', '(?i)intercept'),
        ('Reaction ~ Days + (0 + Days | Subject)', '(?i)intercept'),
    ],
)
def test_random_terms_the_fixed_terms_cannot_lead_with_raise(api: Api, formula: str, term: str):
    df = _sleepstudy(Noise=np.random.default_rng(1).normal(size=120))

    with pytest.raises(ValueError, match=term):
        api.sample(df, formula=formula, n_samples=8, fit_preprocessor=True)


def _fixedEffectRows(summary: str) -> dict[str, str]:
    """The Fixed Effects table of a posteriorSummary, one row per parameter name."""
    lines = summary.splitlines()
    start = lines.index('Fixed Effects:') + 3  # header and separator
    rows = lines[start : lines.index('', start)]
    return {row.split('|')[1].strip(): row for row in rows}


def test_fixed_term_order_does_not_change_the_fit(api: Api):
    df = _sleepstudy(Noise=np.random.default_rng(1).normal(size=120))
    kwargs = dict(n_samples=64, fit_preprocessor=True)

    torch.manual_seed(0)
    first = api.sample(df, formula='Reaction ~ Days + Noise + (1 + Days | Subject)', **kwargs)
    torch.manual_seed(0)
    later = api.sample(df, formula='Reaction ~ Noise + Days + (1 + Days | Subject)', **kwargs)

    assert first.param_names['ffx'] == ['Intercept', 'Days', 'Noise']
    assert later.param_names['ffx'] == ['Intercept', 'Noise', 'Days']
    # both build the same batch internally, so under one seed the draws are equal after
    # permutation, not just within Monte Carlo noise
    perm = [later.param_names['ffx'].index(name) for name in first.param_names['ffx']]
    assert torch.equal(later.proposal.ffx[..., perm], first.proposal.ffx[..., :3])
    for key in ('sigma_rfx', 'rfx'):
        assert torch.equal(getattr(later.proposal, key), getattr(first.proposal, key))

    first_rows = _fixedEffectRows(api.posteriorSummary(first))
    later_rows = _fixedEffectRows(api.posteriorSummary(later))
    assert list(later_rows) == ['Intercept', 'Noise', 'Days']
    assert later_rows == first_rows
    grid = api.plotParameters(later)
    labels = [ax.get_ylabel() for ax in grid.axes[:, 0]]
    plt.close(grid.figure)
    assert labels.index('Noise') < labels.index('Days')
