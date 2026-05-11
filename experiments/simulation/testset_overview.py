"""
Summarize preprocessed test datasets and compare their size structure to generated data.

The default view focuses on continuous-outcome test datasets, which are the
closest real-data analogue for normal-family oracle evaluations.

Usage from repo root:
    uv run python experiments/simulation/testset_overview.py
"""

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as sp_entropy

from metabeta.utils.experiments import DATA_DIR, PREPROCESSED_DATA_DIR


@dataclass
class DatasetSummary:
    source: str
    name: str
    d: int
    q: int | None
    m: int
    n: int
    ns_min: int
    ns_max: int
    ns_mean: float
    ns_std: float
    ns_entropy_ratio: float
    y_type: str = ''
    nu_ffx: np.ndarray | None = None
    tau_ffx: np.ndarray | None = None
    columns: np.ndarray | None = None

    @property
    def mean_group_size(self) -> float:
        return self.n / self.m


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-dir',
        type=Path,
        default=PREPROCESSED_DATA_DIR / 'test',
        help='Directory containing preprocessed real test .npz files.',
    )
    parser.add_argument(
        '--y-type',
        default='continuous',
        choices=['continuous', 'binary', 'count', 'multiclass', 'all'],
        help='Real test-set outcome type to include. Use continuous for normal-family checks.',
    )
    parser.add_argument(
        '--generated-data-ids',
        nargs='+',
        default=['small-n-mixed', 'medium-n-mixed', 'large-n-mixed', 'huge-n-mixed'],
        help='Generated data ids to compare against the real test pool.',
    )
    parser.add_argument(
        '--generated-glob',
        default='train_ep*.npz',
        help='File glob within each generated data directory.',
    )
    parser.add_argument(
        '--max-generated-files',
        type=int,
        default=2,
        help='Maximum files loaded per generated data id; use 0 for all files.',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('experiments/results/testset_overview'),
        help='Directory for overview plots.',
    )
    parser.add_argument('--no-plots', action='store_true', help='Skip PDF plot generation.')
    return parser.parse_args()
# fmt: on


def BambiDefaultPriors(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute bambi-style default Normal priors for fixed effects."""
    d = X.shape[1] + 1
    sy = np.nanstd(y)

    nu_ffx = np.zeros(d)
    tau_ffx = np.zeros(d)
    nu_ffx[0] = np.nanmean(y)
    tau_ffx[0] = 2.5 * sy

    for j in range(X.shape[1]):
        sx = np.nanstd(X[:, j])
        tau_ffx[j + 1] = 2.5 * sy / sx if sx > 1e-12 else 2.5 * sy

    return nu_ffx, tau_ffx


def _entropyRatio(ns: np.ndarray) -> float:
    ns = ns.astype(float, copy=False)
    if len(ns) <= 1:
        return 1.0
    ns_prob = ns / ns.sum()
    return float(sp_entropy(ns_prob) / np.log(len(ns)))


def _scalar(data: dict, key: str) -> int:
    return int(np.asarray(data[key]).item())


def _realYType(data: dict) -> str:
    if 'y_type' not in data:
        return ''
    value = data['y_type']
    return str(value.item() if np.asarray(value).shape == () else value)


def _summaryFromRealFile(path: Path) -> DatasetSummary:
    with np.load(path, allow_pickle=True) as data:
        ns = np.asarray(data['ns'])
        nu_ffx, tau_ffx = BambiDefaultPriors(np.asarray(data['X']), np.asarray(data['y']))
        columns = np.asarray(data.get('columns', np.array([])))
        return DatasetSummary(
            source='real',
            name=path.stem,
            d=_scalar(data, 'd'),
            q=None,
            m=_scalar(data, 'm'),
            n=_scalar(data, 'n'),
            ns_min=int(ns.min()),
            ns_max=int(ns.max()),
            ns_mean=float(ns.mean()),
            ns_std=float(ns.std()),
            ns_entropy_ratio=_entropyRatio(ns),
            y_type=_realYType(data),
            nu_ffx=nu_ffx,
            tau_ffx=tau_ffx,
            columns=columns,
        )


def _summaryFromGeneratedArrays(data_id: str, data: dict, i: int) -> DatasetSummary:
    m = int(data['m'][i])
    ns = np.asarray(data['ns'][i, :m])
    return DatasetSummary(
        source=data_id,
        name=f'{data_id}:{i}',
        d=int(data['d'][i]),
        q=int(data['q'][i]) if 'q' in data else None,
        m=m,
        n=int(data['n'][i]),
        ns_min=int(ns.min()),
        ns_max=int(ns.max()),
        ns_mean=float(ns.mean()),
        ns_std=float(ns.std()),
        ns_entropy_ratio=_entropyRatio(ns),
    )


def CollectReal(test_dir: Path, y_type: str) -> list[DatasetSummary]:
    rows = []
    for path in sorted(test_dir.glob('*.npz')):
        row = _summaryFromRealFile(path)
        if y_type == 'all' or row.y_type == y_type:
            rows.append(row)
    return rows


def _generatedPaths(data_id: str, glob_pattern: str, max_files: int) -> list[Path]:
    paths = sorted((DATA_DIR / data_id).glob(glob_pattern))
    if max_files > 0:
        paths = paths[:max_files]
    return paths


def CollectGenerated(
    data_ids: Iterable[str], glob_pattern: str, max_files: int
) -> list[DatasetSummary]:
    rows = []
    for data_id in data_ids:
        for path in _generatedPaths(data_id, glob_pattern, max_files):
            with np.load(path, allow_pickle=True) as data:
                batch = {k: data[k] for k in data.files if k in ('d', 'q', 'm', 'n', 'ns')}
            rows.extend(
                _summaryFromGeneratedArrays(data_id, batch, i) for i in range(len(batch['n']))
            )
    return rows


def _quantiles(values: np.ndarray) -> str:
    qs = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return ' '.join(f'{q:>8.2f}' for q in qs)


def _safeCorr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def PrintRows(rows: list[DatasetSummary]) -> None:
    header = (
        f"{'dataset':<32} {'d':>3} {'m':>5} {'n':>7}"
        f"  {'n/m':>7} {'ns_min':>6} {'ns_max':>6} {'H_ratio':>7}"
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        print(
            f'{r.name:<32} {r.d:>3} {r.m:>5} {r.n:>7}  '
            f'{r.mean_group_size:>7.1f} {r.ns_min:>6} {r.ns_max:>6} '
            f'{r.ns_entropy_ratio:>7.3f}'
        )


def PrintDistributionSummary(label: str, rows: list[DatasetSummary]) -> None:
    if not rows:
        print(f'\n{label}: no rows')
        return

    print(f'\n--- {label} ({len(rows)} datasets) ---')
    print(f"{'metric':<16} {'min':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'max':>8}")
    for name, values in [
        ('d', np.array([r.d for r in rows], dtype=float)),
        ('q', np.array([r.q for r in rows if r.q is not None], dtype=float)),
        ('m', np.array([r.m for r in rows], dtype=float)),
        ('n', np.array([r.n for r in rows], dtype=float)),
        ('n_per_group', np.array([r.mean_group_size for r in rows], dtype=float)),
        ('ns_max', np.array([r.ns_max for r in rows], dtype=float)),
        ('H_ratio', np.array([r.ns_entropy_ratio for r in rows], dtype=float)),
    ]:
        if len(values):
            print(f'{name:<16} {_quantiles(values)}')

    log_m = np.log10(np.array([r.m for r in rows], dtype=float))
    log_n = np.log10(np.array([r.n for r in rows], dtype=float))
    log_npg = np.log10(np.array([r.mean_group_size for r in rows], dtype=float))
    print(f'corr(log m, log n)       { _safeCorr(log_m, log_n):>8.3f}')
    print(f'corr(log m, log n/m)     { _safeCorr(log_m, log_npg):>8.3f}')


def PrintCoverage(real_rows: list[DatasetSummary], generated_rows: list[DatasetSummary]) -> None:
    if not real_rows or not generated_rows:
        return

    max_m = max(r.m for r in generated_rows)
    max_n = max(r.n for r in generated_rows)
    max_ng = max(r.ns_max for r in generated_rows)
    print('\n--- Real rows outside loaded generated training files ---')
    print(f'loaded generated maxima: m={max_m}, n={max_n}, ns_max={max_ng}')
    outside = [r for r in real_rows if r.m > max_m or r.n > max_n or r.ns_max > max_ng]
    if not outside:
        print('none')
        return
    for r in sorted(outside, key=lambda x: x.n, reverse=True):
        reasons = []
        if r.m > max_m:
            reasons.append(f'm>{max_m}')
        if r.n > max_n:
            reasons.append(f'n>{max_n}')
        if r.ns_max > max_ng:
            reasons.append(f'ns_max>{max_ng}')
        print(f'  {r.name}: d={r.d}, m={r.m}, n={r.n}, ns_max={r.ns_max} ({", ".join(reasons)})')


def PrintPriorSummary(rows: list[DatasetSummary]) -> None:
    print('\n--- Bambi default prior parameters (nu_ffx, tau_ffx) ---')
    for r in rows:
        if r.nu_ffx is None or r.tau_ffx is None:
            continue
        cols = ['intercept'] + list(r.columns if r.columns is not None else [])
        pairs = [f'{c}: N({nu:.2f}, {tau:.2f})' for c, nu, tau in zip(cols, r.nu_ffx, r.tau_ffx)]
        print(f'  {r.name}: {", ".join(pairs)}')


def Plot(
    real_rows: list[DatasetSummary], generated_rows: list[DatasetSummary], outdir: Path
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if generated_rows:
        ax.scatter(
            [r.m for r in generated_rows],
            [r.n for r in generated_rows],
            s=8,
            alpha=0.25,
            label='generated train',
        )
    ax.scatter([r.m for r in real_rows], [r.n for r in real_rows], s=36, label='real test')
    for r in real_rows:
        if r.n > 3000 or r.m > 150:
            ax.annotate(r.name.split('__')[0], (r.m, r.n), fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('m groups')
    ax.set_ylabel('n observations')
    ax.set_title('Joint size distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'joint_size_distribution.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.75, 1.0, 20)
    if generated_rows:
        ax.hist(
            [r.ns_entropy_ratio for r in generated_rows],
            bins=bins,
            alpha=0.55,
            density=True,
            label='generated train',
        )
    ax.hist(
        [r.ns_entropy_ratio for r in real_rows],
        bins=bins,
        alpha=0.75,
        density=True,
        label='real test',
    )
    ax.set_xlabel('group-size entropy ratio')
    ax.set_ylabel('density')
    ax.set_title('Within-dataset group-size balance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'group_entropy_distribution.pdf')
    plt.close(fig)

    print(f'\nPlots saved to {outdir}/')


def main() -> None:
    args = setup()
    real_rows = CollectReal(args.test_dir, y_type=args.y_type)
    generated_rows = CollectGenerated(
        args.generated_data_ids,
        glob_pattern=args.generated_glob,
        max_files=args.max_generated_files,
    )

    print(f'Real test directory: {args.test_dir}')
    print(f'Real y_type filter: {args.y_type}')
    PrintRows(real_rows)
    PrintDistributionSummary('real test pool', real_rows)
    PrintDistributionSummary('generated training pool', generated_rows)
    PrintCoverage(real_rows, generated_rows)
    PrintPriorSummary(real_rows)

    if not args.no_plots:
        Plot(real_rows, generated_rows, args.outdir)


if __name__ == '__main__':
    main()
