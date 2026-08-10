"""Vector output must stay small: point clouds are rasterized, everything else stays vector."""

import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np
from matplotlib import pyplot as plt

from metabeta.plotting.recovery import _plotRecovery
from metabeta.utils.plot import DPI, VECTOR_RASTER_DPI, savePlot


def _recoveryFigure(n: int = 4000, d: int = 3):
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(n, d))
    estimates = targets + rng.normal(scale=0.3, size=(n, d))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    _plotRecovery(
        ax,
        targets=targets,
        estimates=estimates,
        mask=np.ones((n, d), dtype=bool),
        stats={'r': 0.95},
        names=[rf'$\beta_{{{i}}}$' for i in range(d)],
        colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
    )
    return fig, ax


def test_scatter_is_rasterized_but_text_is_not():
    fig, ax = _recoveryFigure()
    try:
        scatters = ax.collections
        assert scatters, 'expected scatter collections'
        assert all(s.get_rasterized() for s in scatters)
        assert not any(t.get_rasterized() for t in ax.texts)
    finally:
        plt.close(fig)


def test_pdf_defaults_to_the_lower_raster_dpi(tmp_path):
    assert VECTOR_RASTER_DPI < DPI
    fig, _ = _recoveryFigure()
    try:
        savePlot(tmp_path, 'lo', ending='pdf')
        savePlot(tmp_path, 'hi', ending='pdf', dpi=DPI)
    finally:
        plt.close(fig)

    lo = (tmp_path / 'lo_latest.pdf').stat().st_size
    hi = (tmp_path / 'hi_latest.pdf').stat().st_size
    assert lo < hi


def test_png_keeps_full_dpi(tmp_path):
    fig, _ = _recoveryFigure(n=200)
    try:
        savePlot(tmp_path, 'raster', ending='png')
    finally:
        plt.close(fig)

    from PIL import Image

    width, _ = Image.open(tmp_path / 'raster_latest.png').size
    # 6 in wide plus tight-bbox padding; a drop to VECTOR_RASTER_DPI would land near 600 px
    assert width > 6 * DPI * 0.8
