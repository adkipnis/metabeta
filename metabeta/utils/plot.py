from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

cmap = plt.get_cmap('tab20')
PALETTE = [mcolors.to_hex(cmap(i)) for i in range(0, cmap.N, 2)]
PALETTE += [mcolors.to_hex(cmap(i)) for i in range(1, cmap.N, 2)]

def savePlot(
        plot_dir: Path, title: str, epoch: int | None = None, ending: str = 'png'
) -> None:
    fname = plot_dir / f'{title}_latest.{ending}'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
    if epoch is not None:
        fname_e = plot_dir / f'{title}_e{epoch}.{ending}'
        plt.savefig(fname_e, bbox_inches='tight', pad_inches=0.15)
