from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes

DPI = 300

cmap = plt.get_cmap('tab20')
PALETTE = [mcolors.to_hex(cmap(i)) for i in range(0, cmap.N, 2)]
PALETTE += [mcolors.to_hex(cmap(i)) for i in range(1, cmap.N, 2)]

INFO = {
    'show_title': True,
    'title_fs': 32,
    'title_pad': 15,
    'ticks_ls': 20,
    'show_x': True,
    'xlabel_fs': 28,
    'xlabel_pad': 10,
    'show_y': True,
    'ylabel_fs': 28,
    'ylabel_pad': 10,
    'despine': False,
    'show_legend': True,
    'legend_fs': 22,
    'legend_ms': 2.5,
    'legend_loc': 'upper left',
    'stats': None,  # dict[str, float]
    'stats_suffix': '',
    'stats_fs': 22,
    'stats_loc_x': 0.69,
    'stats_loc_y': 0.05,
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
        ax.legend(fontsize=fs, markerscale=ms, loc=loc)

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
            ha='center',
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
