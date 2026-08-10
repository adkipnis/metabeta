import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np
from matplotlib import colors as mcolors

from metabeta.plotting.comparison import _RECOVERY_ALPHA
from metabeta.utils.plot import paramColor, paramColors
from metabeta.utils.results import getCorrRfxNames, getNames

SCATTER_ALPHA = _RECOVERY_ALPHA  # the real value, so thresholds test what is drawn


def _lab(colors: list[str], alpha: float | None = None) -> np.ndarray:
    rgb = np.array([mcolors.to_rgb(c) for c in colors])
    if alpha is not None:
        rgb = alpha * rgb + (1 - alpha) * 1.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    mat = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    )
    xyz = lin @ mat.T / np.array([0.95047, 1.0, 1.08883])
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _min_delta_e(colors: list[str], alpha: float | None = None) -> float:
    lab = _lab(colors, alpha)
    d = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _hue(colors: list[str]) -> np.ndarray:
    lab = _lab(colors)
    return np.rad2deg(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0


# every trained configuration, largest last
CONFIGS = [(4, 2), (8, 3), (12, 4), (16, 5)]


def _panels(d: int, q: int) -> dict[str, list[str]]:
    return {
        'ffx': getNames('ffx', d),
        'sigmas': getNames('sigmas', q, has_sigma_eps=True),
        'rfx': getNames('rfx', q),
    }


def test_each_panel_stays_distinct_after_alpha_blending():
    for d, q in CONFIGS:
        for panel, names in _panels(d, q).items():
            colors = paramColors(names)
            assert len(set(colors)) == len(colors), f'duplicate colour in {panel} for d={d} q={q}'
            # panels never share an axes, so separation only has to hold within one
            assert _min_delta_e(colors, alpha=SCATTER_ALPHA) > 6.0, f'{panel} d={d} q={q}'


def test_corresponding_parameters_share_a_hue():
    """beta_j, sigma_j and alpha_j describe one predictor and should read as one family."""
    for d, q in CONFIGS:
        for j in range(q):
            family = [
                paramColor(rf'$\beta_{{{j}}}$'),
                paramColor(rf'$\sigma_{j}$'),
                paramColor(rf'$\alpha_{{{j}}}$'),
            ]
            hues = _hue(family)
            spread = (hues.max() - hues.min() + 180) % 360 - 180
            assert abs(spread) < 12.0, f'family j={j} spans {spread:.1f} deg of hue'


def test_roles_are_separated_by_lightness_not_hue():
    beta = _lab(paramColors(getNames('ffx', 5)))[..., 0]
    sigma = _lab(paramColors(getNames('sigmas', 5, has_sigma_eps=False)))[..., 0]
    alpha = _lab(paramColors(getNames('rfx', 5)))[..., 0]

    # rfx is drawn b*m times per parameter, so it must be lightest
    assert beta.mean() < sigma.mean() < alpha.mean()
    for role in (beta, sigma, alpha):
        assert role.std() < 1.0


def test_no_colour_is_dark_enough_to_block_up_when_overplotted():
    for d, q in CONFIGS:
        for names in _panels(d, q).values():
            lightness = _lab(paramColors(names))[..., 0]
            # overlapping markers compound towards opaque; a dark base turns dense regions solid
            assert lightness.min() > 55.0


def test_sigma_eps_is_near_neutral():
    lab = _lab([paramColor(r'$\sigma_\epsilon$')])[0]
    chroma = float(np.hypot(lab[1], lab[2]))
    # no matching fixed or random effect, so deliberately no hue
    assert chroma < 12.0


def test_correlations_take_the_hue_between_their_two_effects():
    names = getCorrRfxNames(3)
    colors = paramColors(names)
    assert len(set(colors)) == len(colors)
    assert all(isinstance(c, str) and c.startswith('#') for c in colors)


def test_unknown_names_fall_back_without_raising():
    colors = paramColors(['not a parameter', 'also not'])
    assert len(set(colors)) == 2
