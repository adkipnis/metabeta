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
from scipy import stats
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


def test_prior_assigned_by_name_follows_the_term_in_both_orderings(api: Api):
    df = _sleepstudy(Noise=np.random.default_rng(1).normal(size=120))
    kwargs = dict(n_samples=64, fit_preprocessor=True, priors={'fixed': {'Days': {'tau': 0.123}}})

    torch.manual_seed(0)
    first = api.sample(df, formula='Reaction ~ Days + Noise + (1 + Days | Subject)', **kwargs)
    torch.manual_seed(0)
    later = api.sample(df, formula='Reaction ~ Noise + Days + (1 + Days | Subject)', **kwargs)

    for result in (first, later):
        tau = dict(zip(result.param_names['ffx'], result.prior_params['tau_ffx'][0]))
        assert tau['Days'] == pytest.approx(0.123)
        assert tau['Noise'] == tau['Intercept'] != pytest.approx(0.123)
        assert '  Days ~ N(0, 0.123)' in api.posteriorSummary(result, x_scale='standardized')
    perm = [later.param_names['ffx'].index(name) for name in first.param_names['ffx']]
    assert torch.equal(later.proposal.ffx[..., perm], first.proposal.ffx[..., :3])


def _tableRows(summary: str, title: str) -> dict[str, list[float]]:
    """The numbers of one posteriorSummary table, one row per parameter name."""
    lines = summary.splitlines()
    start = lines.index(title) + 3  # header and separator
    rows = [row.split('|')[1:-1] for row in lines[start : lines.index('', start)]]
    return {row[0].strip(): [float(cell) for cell in row[1:]] for row in rows}


def _randomSlopeDraws(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """σ_Intercept, σ_Days and their correlation per draw, on the standardized scale."""
    sigma = result.proposal.sigma_rfx[0].double().numpy()  # (S, q)
    rho = result.proposal.corr_rfx[0, :, 1, 0].double().numpy()  # (S,)
    return sigma[:, 0], sigma[:, 1], rho


def test_default_summary_reports_random_effects_at_days_zero(api: Api):
    df = _sleepstudy()
    result = api.sample(
        df, formula='Reaction ~ Days + (1 + Days | Subject)', n_samples=64, fit_preprocessor=True
    )

    # b0 + b1 (x - mu) / s = (b0 - b1 mu / s) + (b1 / s) x, so per draw the covariance of
    # (b0', b1') follows from Var(b0 - c b1), Var(b1) / s^2 and Cov(b0 - c b1, b1 / s), c = mu / s
    mu, s = df['Days'].mean(), df['Days'].std(ddof=0)
    c = mu / s
    sd0, sd1, rho = _randomSlopeDraws(result)
    sd0_orig = np.sqrt(sd0**2 - 2 * c * rho * sd0 * sd1 + c**2 * sd1**2)
    sd1_orig = sd1 / s
    rho_orig = (rho * sd0 * sd1 - c * sd1**2) / s / (sd0_orig * sd1_orig)

    summary = api.posteriorSummary(result)
    sds = _tableRows(summary, 'Standard Deviations:')
    corr = _tableRows(summary, 'Correlations:')
    assert sds['Intercept'][0] == pytest.approx(sd0_orig.mean(), abs=1e-3)
    assert sds['Days'][0] == pytest.approx(sd1.mean() / s, abs=1e-3)
    assert corr['Days'][0] == pytest.approx(rho_orig.mean(), abs=1e-3)
    assert corr['Intercept'][1] == corr['Days'][0]
    tau = result.prior_params['tau_rfx'][0, 1] * df['Reaction'].std(ddof=0) / s
    assert f'  σ_Days ~ HN({tau:.4g})' in summary


def test_random_intercept_sd_does_not_depend_on_the_predictor_scale(api: Api):
    result = api.sample(
        _sleepstudy(),
        formula='Reaction ~ Days + (1 | Subject)',
        n_samples=64,
        fit_preprocessor=True,
    )

    original = _tableRows(api.posteriorSummary(result), 'Standard Deviations:')
    standardized = _tableRows(
        api.posteriorSummary(result, x_scale='standardized'), 'Standard Deviations:'
    )
    assert list(original) == ['Intercept', 'Residual']
    assert original == standardized


def _rfxRows(rfx_summary: str, name: str) -> tuple[dict[str, list[float]], float]:
    """One component's table of an rfxSummary, one row per group, and its SD ratio."""
    lines = rfx_summary.splitlines()
    start = lines.index(f'{name}:') + 3  # header and separator
    end = next(i for i in range(start, len(lines)) if lines[i].startswith('SD ratio = '))
    rows = [row.split('|')[1:-1] for row in lines[start:end]]
    ratio = float(lines[end].removeprefix('SD ratio = '))
    return {row[0].strip(): [float(cell) for cell in row[1:]] for row in rows}, ratio


def test_standardized_summaries_print_the_standardized_draws(api: Api):
    result = api.sample(
        _sleepstudy(),
        formula='Reaction ~ Days + (1 + Days | Subject)',
        n_samples=64,
        fit_preprocessor=True,
    )
    sd0, sd1, rho = _randomSlopeDraws(result)
    rfx = result.proposal.rfx[0].double().numpy()  # (m, S, q)

    summary = api.posteriorSummary(result, x_scale='standardized')
    sds = _tableRows(summary, 'Standard Deviations:')
    assert sds['Intercept'][0] == pytest.approx(sd0.mean(), abs=1e-3)
    assert sds['Days'][0] == pytest.approx(sd1.mean(), abs=1e-3)
    assert _tableRows(summary, 'Correlations:')['Days'][0] == pytest.approx(rho.mean(), abs=1e-3)

    rfx_summary = api.rfxSummary(result, x_scale='standardized')
    for k, (name, sd) in enumerate([('Intercept', sd0), ('Days', sd1)]):
        rows, ratio = _rfxRows(rfx_summary, name)
        means = rfx[:, :, k].mean(axis=1)  # (m,)
        assert [row[1] for row in rows.values()] == pytest.approx(means, abs=1e-3)
        assert ratio == pytest.approx(means.std(ddof=1) / sd.mean(), abs=0.01)


def test_default_rfx_summary_reports_group_effects_at_days_zero(api: Api):
    df = _sleepstudy()
    result = api.sample(
        df, formula='Reaction ~ Days + (1 + Days | Subject)', n_samples=64, fit_preprocessor=True
    )
    mu, s = df['Days'].mean(), df['Days'].std(ddof=0)
    rfx = result.proposal.rfx[0].double().numpy()  # (m, S, q)
    b0 = rfx[..., 0] - mu / s * rfx[..., 1]  # (m, S), the group's deviation at Days = 0
    b1 = rfx[..., 1] / s  # (m, S), per day
    sd_rows = _tableRows(api.posteriorSummary(result), 'Standard Deviations:')

    rfx_summary = api.rfxSummary(result)
    for name, b in [('Intercept', b0), ('Days', b1)]:
        rows, ratio = _rfxRows(rfx_summary, name)
        means = b.mean(axis=1)  # (m,)
        assert [row[1] for row in rows.values()] == pytest.approx(means, abs=1e-3)
        assert [row[2] for row in rows.values()] == pytest.approx(b.std(axis=1, ddof=1), abs=1e-3)
        assert ratio == pytest.approx(means.std(ddof=1) / sd_rows[name][0], abs=0.01)


def test_integer_group_ids_label_their_own_rows(api: Api):
    df = _sleepstudy()
    df['Subject'] = np.repeat(np.arange(12), 10)  # ids 0-11; as strings '10' sorts before '2'
    df.loc[df['Subject'] == 10, 'Reaction'] += 300

    result = api.sample(
        df, formula='Reaction ~ Days + (1 | Subject)', n_samples=64, fit_preprocessor=True
    )

    rows, _ = _rfxRows(api.rfxSummary(result), 'Intercept')
    assert list(rows) == [str(i) for i in range(12)]
    assert max(rows, key=lambda group: rows[group][1]) == '10'


def _gridDraws(grid) -> np.ndarray:
    """The draws a plotParameters grid shows, (S, D), read from its upper scatter cells."""
    n = grid.axes.shape[0]
    first = grid.axes[0, 1].collections[0].get_offsets()[:, 1]
    rest = [grid.axes[0, j].collections[0].get_offsets()[:, 0] for j in range(1, n)]
    return np.column_stack([first, *rest]).astype(float)


def _priorCurve(grid, i: int) -> tuple[np.ndarray, np.ndarray]:
    """The prior overlay (x, density) drawn on the grid's i-th diagonal cell."""
    ax = grid.axes[i, i]
    twins = [a for a in grid.figure.axes if a is not ax and a.bbox.bounds == ax.bbox.bounds]
    (line,) = [line for a in twins for line in a.get_lines()]
    return np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)


def test_parameter_plot_draws_random_effects_and_slope_prior_at_days_zero(api: Api):
    df = _sleepstudy()
    result = api.sample(
        df, formula='Reaction ~ Days + (1 + Days | Subject)', n_samples=64, fit_preprocessor=True
    )
    mu, s = df['Days'].mean(), df['Days'].std(ddof=0)
    c = mu / s
    sd0, sd1, rho = _randomSlopeDraws(result)
    sd0_orig = np.sqrt(sd0**2 - 2 * c * rho * sd0 * sd1 + c**2 * sd1**2)
    rho_orig = (rho * sd0 * sd1 - c * sd1**2) / s / (sd0_orig * sd1 / s)

    grid = api.plotParameters(result)
    draws = _gridDraws(grid)  # Intercept, Days, σ_Intercept, σ_Days, σ_ε, ρ
    x, density = _priorCurve(grid, 3)
    plt.close(grid.figure)

    assert draws.shape[1] == 6
    np.testing.assert_allclose(draws[:, 2], sd0_orig, rtol=1e-4)
    np.testing.assert_allclose(draws[:, 3], sd1 / s, rtol=1e-4)
    np.testing.assert_allclose(draws[:, 5], rho_orig, rtol=1e-4, atol=1e-5)
    tau = result.prior_params['tau_rfx'][0, 1] * df['Reaction'].std(ddof=0) / s
    np.testing.assert_allclose(density, stats.halfnorm.pdf(x, scale=tau), rtol=1e-5)


def test_exponential_sd_prior_is_drawn_as_an_exponential(api: Api):
    df = _sleepstudy()
    result = api.sample(
        df,
        formula='Reaction ~ Days + (1 + Days | Subject)',
        n_samples=64,
        fit_preprocessor=True,
        priors={'random_sd': {'family': 'exponential', 'Days': {'tau': 0.7}}},
    )

    grid = api.plotParameters(result)
    x, density = _priorCurve(grid, 3)  # σ_Days
    plt.close(grid.figure)

    scale = 0.7 * df['Reaction'].std(ddof=0) / df['Days'].std(ddof=0)
    np.testing.assert_allclose(density, stats.expon.pdf(x, scale=scale), rtol=1e-5)


def _cdfAt(curve: tuple[np.ndarray, np.ndarray], points: np.ndarray) -> np.ndarray:
    """The CDF of a density drawn on a grid, at the given points."""
    x, density = curve
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(x))])
    return np.interp(points, x, cdf / cdf[-1])


def test_prior_overlays_without_closed_form_match_the_pushforward(api: Api):
    df = _sleepstudy()
    result = api.sample(
        df, formula='Reaction ~ Days + (1 + Days | Subject)', n_samples=64, fit_preprocessor=True
    )

    grid = api.plotParameters(result)
    curves = {'sd0': _priorCurve(grid, 2), 'rho': _priorCurve(grid, 5)}
    plt.close(grid.figure)

    # the prior pushed through A by hand: half-normal SDs and, for q = 2, an LKJ(eta)
    # correlation 2 Beta(eta, eta) - 1
    rng = np.random.default_rng(1)
    n = 200_000
    tau = result.prior_params['tau_rfx'][0, :2] * df['Reaction'].std(ddof=0)
    eta = float(result.prior_params['eta_rfx'][0])
    sd0, sd1 = np.abs(rng.normal(size=(2, n))) * tau[:, None]
    rho = 2 * rng.beta(eta, eta, n) - 1
    c, s = df['Days'].mean() / df['Days'].std(ddof=0), df['Days'].std(ddof=0)
    sd0_orig = np.sqrt(sd0**2 - 2 * c * rho * sd0 * sd1 + c**2 * sd1**2)
    rho_orig = (rho * sd0 * sd1 - c * sd1**2) / s / (sd0_orig * sd1 / s)

    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    for key, draws in [('sd0', sd0_orig), ('rho', rho_orig)]:
        cdf = _cdfAt(curves[key], np.quantile(draws, levels))
        np.testing.assert_allclose(cdf, levels, atol=0.015, err_msg=key)


def _warningsOf(call) -> tuple[object, list[warnings.WarningMessage]]:
    """The result of `call()` and every warning it emits."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return result, caught


def test_low_ess_warning_suggests_more_draws_and_points_at_the_call(api: Api):
    _, caught = _warningsOf(
        lambda: api.sample(
            _sleepstudy(),
            formula='Reaction ~ Days + (1 + Days | Subject)',
            n_samples=64,
            fit_preprocessor=True,
        )
    )

    (ess,) = [w for w in caught if 'effective sample size' in str(w.message)]
    assert 'n_samples' in str(ess.message)
    assert 'do not fix' not in str(ess.message)
    assert ess.filename == __file__
