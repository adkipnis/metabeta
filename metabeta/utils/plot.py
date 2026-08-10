from pathlib import Path
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

DPI = 300

# Resolution for rasterized artists inside vector output. Comparison figures are ~30 in wide and
# land on a page at ~7 in, so this is still ~430 dpi in print while keeping the file an order of
# magnitude smaller than embedding the point clouds at DPI.
VECTOR_RASTER_DPI = 100

# tab20's ordering kept verbatim, then extended: tab20 stops at 20 but the largest models label 27
# series, so it used to wrap and repeat colours. Additions sit at tab20's pastel lightness (L* 78)
# so none reads darker than its neighbours. Used by warmfit and as a fallback; parameter panels use
# paramColors below.
PALETTE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#aec7e8',
    '#ffbb78',
    '#98df8a',
    '#ff9896',
    '#c5b0d5',
    '#c49c94',
    '#f7b6d2',
    '#c7c7c7',
    '#dbdb8d',
    '#9edae5',
    '#6fd4b2',
    '#d8be8f',
    '#6ad96a',
    '#a7cc8f',
    '#ecaced',
    '#93cdb9',
    '#afcd6a',
    '#dfb7c0',
    '#6dd696',
    '#e1baa2',
    '#71d1ce',
    '#90d0a3',
]


# Parameter colours. Panels colour a contiguous run of parameters and never share an axes, so
# separation only has to hold within a panel (<=16 colours), not across all 27. That slack pays for
# encoding meaning: hue identifies the predictor, so beta_j / sigma_j / alpha_j are three shades of
# one colour, and lightness identifies the role. rho_ij takes the hue between its two effects;
# sigma_eps has no matching predictor and is near-neutral.
#
# role -> (L*, chroma fraction, alternate chroma fraction). Chroma is a fraction of what sRGB can
# reach at that lightness and hue, never an absolute target: the ceiling swings from ~90 in the
# pinks to ~36 in the cyans, and a fixed target clips the cyans into each other.
_ROLE_STYLE = {
    'ffx': (66.0, 0.95, 0.65),
    'sigmas': (73.0, 0.95, 0.65),
    'rfx': (78.0, 0.95, 0.65),
    'corr': (70.0, 0.95, 0.65),
}
_SIGMA_EPS_STYLE = (73.0, 0.10)
_SIGMA_EPS_HUE = 80.0

# Chroma alternates by position around the hue circle, not by index: bit-reversal sends consecutive
# indices to opposite sides, so index parity would give hue-neighbours matching chroma.
_HUE_SLOTS = 16

_D65 = np.array([0.95047, 1.0, 1.08883])
_XYZ2RGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def _lab2rgb(lab: np.ndarray) -> np.ndarray:
    """CIE-Lab to sRGB, unclipped so gamut violations stay detectable."""
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    e, k = 216 / 24389, 24389 / 27
    f = np.array([fx, fy, fz])
    xyz = np.where(f**3 > e, f**3, (116.0 * f - 16.0) / k) * _D65
    lin = _XYZ2RGB @ xyz
    return np.where(
        lin <= 0.0031308,
        12.92 * lin,
        1.055 * np.power(np.clip(lin, 0.0, None), 1 / 2.4) - 0.055,
    )


def _maxChroma(lightness: float, hue_deg: float) -> float:
    """Largest in-gamut C* at this lightness and hue."""
    rad = np.deg2rad(hue_deg)
    lo, hi = 0.0, 150.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        rgb = _lab2rgb(np.array([lightness, mid * np.cos(rad), mid * np.sin(rad)]))
        if np.all(rgb >= -1e-6) and np.all(rgb <= 1 + 1e-6):
            lo = mid
        else:
            hi = mid
    return lo


def _lch2hex(lightness: float, chroma: float, hue_deg: float) -> str:
    rad = np.deg2rad(hue_deg)
    rgb = _lab2rgb(np.array([lightness, chroma * np.cos(rad), chroma * np.sin(rad)]))
    return mcolors.to_hex(np.clip(rgb, 0.0, 1.0))


def _huePosition(index: int) -> float:
    """Bit-reversed position in [0, 1): every prefix stays well spread, so the first q hues are
    far apart whatever q is, while the full run of d lands on an even spacing."""
    fraction, denominator, n = 0.0, 1.0, index
    while n > 0:
        denominator /= 2.0
        fraction += denominator * (n % 2)
        n //= 2
    return fraction


def _hue(index: int) -> float:
    return 360.0 * _huePosition(index)


def _hueRank(index: int) -> int:
    """Position of this hue around the circle."""
    return int(round(_huePosition(index) * _HUE_SLOTS))


_BETA_RE = re.compile(r'\\beta_\{(\d+)\}')
_SIGMA_RE = re.compile(r'\\sigma_(\d+)')
_ALPHA_RE = re.compile(r'\\alpha_\{(\d+)\}')
_RHO_RE = re.compile(r'\\rho_\{(\d)(\d)\}')
_SIGMA_EPS = r'\\sigma_\\epsilon'


def paramColor(name: str, fallback: int = 0) -> str:
    """Colour for a parameter, derived from its name, so every panel agrees without bookkeeping."""
    if re.search(_SIGMA_EPS, name):
        lightness, fraction = _SIGMA_EPS_STYLE
        return _lch2hex(lightness, fraction * _maxChroma(lightness, _SIGMA_EPS_HUE), _SIGMA_EPS_HUE)
    for regex, role in ((_BETA_RE, 'ffx'), (_SIGMA_RE, 'sigmas'), (_ALPHA_RE, 'rfx')):
        match = regex.search(name)
        if match:
            index = int(match.group(1))
            lightness, full, alternate = _ROLE_STYLE[role]
            hue = _hue(index)
            fraction = full if _hueRank(index) % 2 == 0 else alternate
            return _lch2hex(lightness, fraction * _maxChroma(lightness, hue), hue)
    match = _RHO_RE.search(name)
    if match:
        first, second = _hue(int(match.group(1))), _hue(int(match.group(2)))
        # circular midpoint: a correlation reads as a blend of the two effects it relates
        mid = (
            float(
                np.rad2deg(
                    np.angle(np.exp(1j * np.deg2rad(first)) + np.exp(1j * np.deg2rad(second)))
                )
            )
            % 360.0
        )
        lightness, full, _ = _ROLE_STYLE['corr']
        return _lch2hex(lightness, full * _maxChroma(lightness, mid), mid)
    return PALETTE[fallback % len(PALETTE)]


def paramColors(names: list[str]) -> list[str]:
    """Colours for a list of parameter names, in order."""
    return [paramColor(name, fallback=i) for i, name in enumerate(names)]


# fallback marker size (points) when a handle carries no size of its own
_PROXY_MS = 10.0


def legendProxy(handle, label: str):
    """Opaque stand-in for a plotted artist, or the artist itself if not applicable.

    Series are drawn translucent, and the legend inherits that alpha and washes the swatch out.
    ``fill_between`` bands pass through untouched: their translucency is meaningful.
    """
    if not isinstance(handle, (PathCollection, Line2D)):
        return handle

    if isinstance(handle, PathCollection):
        face = np.asarray(handle.get_facecolor())
        color = tuple(face[0][:3]) if face.size else (0.0, 0.0, 0.0)
        sizes = np.asarray(handle.get_sizes())
        # scatter sizes are areas in pt^2; Line2D markersize is a diameter in pt
        ms = float(np.sqrt(sizes[0])) if sizes.size else _PROXY_MS
    else:
        color = mcolors.to_rgb(handle.get_color())
        ms = (
            float(handle.get_markersize())
            if handle.get_marker() not in (None, 'None')
            else _PROXY_MS
        )

    return Line2D([], [], marker='o', linestyle='', markersize=ms, color=color, label=label)


INFO = {
    'show_title': True,
    'title_fs': 33,
    'title_pad': 15,
    'ticks_ls': 20,
    'show_x': True,
    'xlabel_fs': 33,
    'xlabel_pad': 10,
    'show_y': True,
    'ylabel_fs': 33,
    'ylabel_pad': 10,
    'despine': False,
    'show_legend': True,
    'legend_fs': 24,
    'legend_ms': 2.5,
    'legend_loc': 'upper left',
    'stats': None,  # dict[str, float]
    'stats_suffix': '',
    'stats_fs': 22,
    'stats_loc_x': 0.69,
    'stats_loc_y': 0.05,
    'stats_ha': 'center',  # anchor side of (stats_loc_x, stats_loc_y); 'right' insets the box
    'stats_box': True,
    'grid_alpha': 0.8,
}


def niceify(ax: Axes, info: dict[str, float | str | int]) -> None:
    info = INFO | info

    # ticks
    if ticks_ls := info['ticks_ls']:
        ax.tick_params(axis='both', labelsize=ticks_ls)
    else:
        ax.tick_params(axis='x', labelcolor='w', size=1)
        ax.tick_params(axis='y', labelcolor='w', size=1)

    # grid
    grid_alpha = info['grid_alpha']
    if grid_alpha != 1.0:
        ax.grid(True, alpha=grid_alpha)

    # title
    if info['show_title'] and info.get('title') is not None:
        title = str(info['title'])
        fs = int(info['title_fs'])
        pad = int(info['title_pad'])
        ax.set_title(title, fontsize=fs, pad=pad)

    # x label
    if not info['show_x']:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelcolor='w', size=1)
    elif info.get('xlabel') is not None:
        label = str(info['xlabel'])
        fs = int(info['xlabel_fs'])
        pad = int(info['xlabel_pad'])
        ax.set_xlabel(label, fontsize=fs, labelpad=pad)

    # y label
    if info['show_y'] and info.get('ylabel') is not None:
        label = str(info['ylabel'])
        fs = int(info['ylabel_fs'])
        pad = int(info['ylabel_pad'])
        ax.set_ylabel(label, fontsize=fs, labelpad=pad)

    # despine
    if info['despine']:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # legend
    if info['show_legend']:
        fs = info['legend_fs']
        ms = info['legend_ms']
        loc = info['legend_loc']
        handles, labels = ax.get_legend_handles_labels()
        proxies = [legendProxy(handle, label) for handle, label in zip(handles, labels)]
        ax.legend(proxies, labels, fontsize=fs, markerscale=ms, loc=loc)

    # stats
    if (stats := info['stats']) is not None:
        suffix = info['stats_suffix']
        txt = [f'{k} = {v:.3f}{suffix}' for k, v in stats.items()]  # type: ignore
        fs = int(info['stats_fs'])
        x_loc = float(info['stats_loc_x'])
        y_loc = float(info['stats_loc_y'])
        bbox = None
        if info['stats_box']:
            bbox = dict(
                boxstyle='round',
                facecolor=(1, 1, 1, 0.7),
                edgecolor=(0, 0, 0, 0.15),
            )
        ax.text(
            x_loc,
            y_loc,
            '\n'.join(txt),
            transform=ax.transAxes,
            ha=str(info['stats_ha']),
            # keep the lines centred inside the box regardless of how the box is anchored
            ma='center',
            va='bottom',
            fontsize=fs,
            bbox=bbox,
        )


def savePlot(
    plot_dir: Path,
    title: str,
    epoch: int | None = None,
    ending: str = 'png',
    dpi: float | None = None,
) -> Path:
    # vector formats only use dpi for rasterized artists, so a lower value shrinks them alone
    if dpi is None:
        dpi = VECTOR_RASTER_DPI if ending in ('pdf', 'svg', 'eps') else DPI
    fname = plot_dir / f'{title}_latest.{ending}'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.15, dpi=dpi)
    if epoch is not None:
        fname_e = plot_dir / f'{title}_e{epoch}.{ending}'
        plt.savefig(fname_e, bbox_inches='tight', pad_inches=0.15, dpi=dpi)
        return fname_e
    return fname
