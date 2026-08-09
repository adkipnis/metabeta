import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from metabeta.utils.plot import PALETTE, legendProxy

# largest configuration currently trained: 16 ffx + 5 sigma_rfx + sigma_eps + 5 rfx
MAX_SERIES = 27
SCATTER_ALPHA = 0.35


def _srgb2lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, float)
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
    rgb = np.array([mcolors.to_rgb(c) for c in colors])
    if alpha is not None:
        rgb = alpha * rgb + (1 - alpha) * 1.0
    lab = _srgb2lab(rgb)
    d = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def test_palette_covers_the_largest_model_without_repeating():
    assert len(PALETTE) >= MAX_SERIES
    assert len(set(PALETTE)) == len(PALETTE)


def test_palette_stays_distinct_once_blended_at_the_scatter_alpha():
    used = PALETTE[:MAX_SERIES]
    # the palette is only ever seen composited onto white, so that is where it has to hold up.
    # the muted pool measures 5.3 blended / 14.2 opaque; a louder one scores better but reads as
    # neon, so restraint was preferred. these floors just guard against regressing towards the
    # old tab20 ordering, which repeated colours outright and scored 0.
    assert _min_delta_e(used, alpha=SCATTER_ALPHA) > 4.5
    assert _min_delta_e(used) > 12.0


def test_consecutive_colours_are_saliently_different():
    """Panels colour contiguous slices, so neighbours are what actually sit side by side."""
    used = PALETTE[:MAX_SERIES]
    adjacent = [_min_delta_e([a, b], alpha=SCATTER_ALPHA) for a, b in zip(used, used[1:])]

    # the ordering is a bottleneck-maximising path: unordered selection bottoms out near 8,
    # this ordering holds every consecutive step above ~19 even after alpha blending
    assert min(adjacent) > 15.0
    assert min(_min_delta_e([a, b]) for a, b in zip(used, used[1:])) > 40.0


def test_scatter_proxy_is_opaque_and_keeps_colour_and_size():
    fig, ax = plt.subplots()
    handle = ax.scatter([0, 1], [0, 1], color='#b924f2', s=70, alpha=SCATTER_ALPHA, label='b0')

    proxy = legendProxy(handle, 'b0')

    assert isinstance(proxy, Line2D)
    assert proxy.get_alpha() in (None, 1.0)
    assert mcolors.to_hex(proxy.get_color()) == '#b924f2'
    # scatter carries an area in pt^2; the proxy needs the corresponding diameter
    assert proxy.get_markersize() == pytest.approx(np.sqrt(70))
    plt.close(fig)


def test_proxy_does_not_disturb_the_plotted_transparency():
    fig, ax = plt.subplots()
    handle = ax.scatter([0, 1], [0, 1], color='#30e22e', s=70, alpha=SCATTER_ALPHA, label='b1')

    legendProxy(handle, 'b1')

    # the whole point is an opaque legend without touching the points themselves
    assert float(np.asarray(handle.get_facecolor())[0][3]) == pytest.approx(SCATTER_ALPHA)
    plt.close(fig)


def test_line_proxy_keeps_colour():
    fig, ax = plt.subplots()
    (handle,) = ax.plot([0, 1], [0, 1], color='#e61a41', alpha=0.75, label='cov')

    proxy = legendProxy(handle, 'cov')

    assert isinstance(proxy, Line2D)
    assert mcolors.to_hex(proxy.get_color()) == '#e61a41'
    assert proxy.get_alpha() in (None, 1.0)
    plt.close(fig)


def test_band_handles_pass_through_untouched():
    fig, ax = plt.subplots()
    handle = ax.fill_between([0, 1], [0, 0], [1, 1], color='grey', alpha=0.15, label='95% CB')

    proxy = legendProxy(handle, '95% CB')

    # SBC credible bands are meant to read as translucent areas, not as opaque markers
    assert proxy is handle
    assert isinstance(proxy, PolyCollection)
    plt.close(fig)


def test_right_legend_fits_inside_the_figure_for_the_largest_model():
    from matplotlib.lines import Line2D as _Line2D

    from metabeta.plotting.comparison import (
        _PLOT_RIGHT_MARGIN,
        _RIGHT_LEGEND_FONTSIZE,
        _RIGHT_LEGEND_X,
    )
    from metabeta.utils.plot import DPI

    nrows, ncols = 3, 5
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=DPI, squeeze=False)
    for ax in axs.flat:
        ax.set_box_aspect(1)
    handles = [
        _Line2D(
            [],
            [],
            marker='o',
            ls='',
            markersize=np.sqrt(70),
            color=PALETTE[i % len(PALETTE)],
            label=f'p{i}',
        )
        for i in range(MAX_SERIES)
    ]
    legend = fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc='center left',
        bbox_to_anchor=(_RIGHT_LEGEND_X, 0.5),
        fontsize=_RIGHT_LEGEND_FONTSIZE,
        markerscale=2.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, _PLOT_RIGHT_MARGIN, 1.0))
    fig.canvas.draw()

    height = legend.get_window_extent(fig.canvas.get_renderer()).height
    # a taller legend than the panel stack would grow the canvas under bbox_inches='tight'
    assert height <= fig.get_figheight() * fig.dpi
    plt.close(fig)
