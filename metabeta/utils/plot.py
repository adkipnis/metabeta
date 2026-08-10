from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

DPI = 300

# The largest models label 27 series (16 ffx + 5 sigma_rfx + sigma_eps + 5 rfx), so a palette has
# to stay distinct that far out; the previous tab20-derived one held only 20 and silently repeated
# colours beyond that. Colours were picked by greedy farthest-point search in CIE-Lab *after*
# compositing onto white at the scatter alpha, which is the only form the eye ever sees, over a
# deliberately muted pool (HUSL lightness 40/54/68, saturation 42-60) for mean chroma 37 — well
# under tab20's 45, so the figures stay restrained in print.
#
# Order matters as much as membership: every panel colours a *contiguous* slice (see
# _paletteSlice), so neighbouring indices are the ones seen side by side, and a merely
# globally-distinct palette can still seat two near-twins together. The order below maximises the
# weakest consecutive step (a bottleneck Hamiltonian path over the same distance matrix), lifting
# adjacent dE from 8 to 19.5 blended / 57.6 opaque. Keep entries in this order; reshuffling
# preserves the set but throws away that property.
#
# Global minimum pairwise dE is 5.3 blended / 14.2 opaque, against 0.0 for what it replaces. A
# louder pool scores better globally but reads as neon; restraint was chosen deliberately.
# Regenerate with the same two-stage search if the series count ever outgrows this list.
PALETTE = [
    '#c26372',
    '#6bb1a1',
    '#8a4288',
    '#828260',
    '#b25dc7',
    '#d19593',
    '#4657a8',
    '#785739',
    '#c8579a',
    '#46644e',
    '#d38adb',
    '#558891',
    '#a13a40',
    '#a09edb',
    '#c06750',
    '#5c80c1',
    '#d79568',
    '#4c5e7c',
    '#da5152',
    '#72aad4',
    '#d45374',
    '#518b6e',
    '#b56693',
    '#7cb083',
    '#9770b4',
    '#b4a367',
    '#884966',
    '#67b567',
    '#cd92bc',
    '#68894f',
    '#6749b5',
    '#8b4b46',
]

# fallback marker size (points) when a handle carries no size of its own
_PROXY_MS = 10.0


def legendProxy(handle, label: str):
    """Return an opaque stand-in for a plotted artist, or the artist itself if not applicable.

    Series are drawn semi-transparent so that overlapping points stay readable, but the legend
    inherits that alpha and washes the swatch out, which is exactly what makes a colour hard to
    match back to the plot. The proxy keeps the original colour and marker size at full opacity
    and leaves the plotted alpha alone. Artists that are not scatter/line handles (e.g. the
    ``fill_between`` bands in the SBC panels) are passed through untouched, since their
    translucency is meaningful rather than incidental.
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


def savePlot(plot_dir: Path, title: str, epoch: int | None = None, ending: str = 'png') -> Path:
    fname = plot_dir / f'{title}_latest.{ending}'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
    if epoch is not None:
        fname_e = plot_dir / f'{title}_e{epoch}.{ending}'
        plt.savefig(fname_e, bbox_inches='tight', pad_inches=0.15)
        return fname_e
    return fname
