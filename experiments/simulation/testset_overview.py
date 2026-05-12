"""
Summarize preprocessed test datasets and compare their size structure to generated data.

The default view focuses on continuous-outcome test datasets, which are the
closest real-data analogue for normal-family oracle evaluations.

Usage from repo root:
    uv run python experiments/simulation/testset_overview.py
"""

import argparse
import gc
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


@dataclass
class DesignSummary:
    source: str
    name: str
    d: int
    q: int | None
    m: int
    n: int
    x_rank: int
    x_rank_gap: int
    x_condition: float
    min_group_z_rank: int | None = None
    min_group_z_df: int | None = None


_Y_TYPE_TO_FAMILY_LETTER = {'continuous': 'n', 'binary': 'b', 'count': 'p'}


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
        default=None,
        help='Generated data ids to compare against the real test pool. '
        'Defaults to small/medium/large/huge-{family}-mixed derived from --y-type.',
    )
    parser.add_argument(
        '--generated-glob',
        nargs='+',
        default=['train_ep*.npz', 'valid.npz', 'test.npz'],
        help='File glob(s) within each generated data directory. '
        'Multiple globs are unioned; files matched by any glob are included.',
    )
    parser.add_argument(
        '--max-generated-files',
        type=int,
        default=0,
        help='Maximum files loaded per generated data id; use 0 for all files.',
    )
    parser.add_argument(
        '--max-generated-design-files',
        type=int,
        default=1,
        help='Maximum generated files per data id used for design rank/condition checks.',
    )
    parser.add_argument(
        '--max-generated-design-rows',
        type=int,
        default=64,
        help='Maximum generated datasets per file used for design rank/condition checks.',
    )
    parser.add_argument(
        '--design-seed',
        type=int,
        default=0,
        help='Seed for selecting generated datasets for design checks.',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('experiments/results/testset_overview'),
        help='Directory for overview plots.',
    )
    parser.add_argument('--no-plots', action='store_true', help='Skip PDF plot generation.')
    args = parser.parse_args()
    if args.generated_data_ids is None:
        letter = _Y_TYPE_TO_FAMILY_LETTER.get(args.y_type, 'n')
        args.generated_data_ids = [
            f'{s}-{letter}-{ds}' for s in ('small', 'medium', 'large', 'huge')
            for ds in ('mixed', 'sampled')
        ]
    return args
# fmt: on


def BambiDefaultPriors(
    X: np.ndarray, y: np.ndarray, y_type: str = 'continuous'
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bambi-style default Normal priors for fixed effects.

    For binary outcomes bambi operates on the logit scale with fixed-width
    priors (intercept: 1.5, slopes: 1.0). For continuous outcomes priors are
    scaled by sd(y)/sd(x_j).
    """
    d = X.shape[1] + 1
    nu_ffx = np.zeros(d)
    tau_ffx = np.zeros(d)

    if y_type == 'binary':
        tau_ffx[0] = 1.5
        tau_ffx[1:] = 1.0
    else:
        sy = np.nanstd(y)
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
        y_type_str = _realYType(data)
        nu_ffx, tau_ffx = BambiDefaultPriors(
            np.asarray(data['X']), np.asarray(data['y']), y_type=y_type_str
        )
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


def _standardizedCondition(X: np.ndarray) -> float:
    """Condition number after dropping constant columns and standardizing scale."""
    std = np.nanstd(X, axis=0)
    keep = np.isfinite(std) & (std > 1e-12)
    if int(keep.sum()) <= 1:
        return 1.0
    Xs = X[:, keep]
    Xs = Xs - np.nanmean(Xs, axis=0, keepdims=True)
    Xs = Xs / np.nanstd(Xs, axis=0, keepdims=True)
    try:
        return float(np.linalg.cond(Xs))
    except np.linalg.LinAlgError:
        return float('inf')


def _designSummary(
    source: str,
    name: str,
    X: np.ndarray,
    groups: np.ndarray,
    d: int,
    q: int | None,
) -> DesignSummary:
    X = np.asarray(X[:, :d], dtype=float)
    groups = np.asarray(groups)
    rank = int(np.linalg.matrix_rank(X))

    min_group_z_rank = None
    min_group_z_df = None
    if q is not None:
        ranks = []
        dfs = []
        for group in np.unique(groups):
            idx = groups == group
            Zg = X[idx, :q]
            ranks.append(int(np.linalg.matrix_rank(Zg)))
            dfs.append(int(idx.sum()) - q)
        min_group_z_rank = min(ranks) if ranks else None
        min_group_z_df = min(dfs) if dfs else None

    return DesignSummary(
        source=source,
        name=name,
        d=d,
        q=q,
        m=int(len(np.unique(groups))),
        n=int(len(X)),
        x_rank=rank,
        x_rank_gap=d - rank,
        x_condition=_standardizedCondition(X),
        min_group_z_rank=min_group_z_rank,
        min_group_z_df=min_group_z_df,
    )


def CollectReal(test_dir: Path, y_type: str) -> list[DatasetSummary]:
    rows = []
    for path in sorted(test_dir.glob('*.npz')):
        row = _summaryFromRealFile(path)
        if y_type == 'all' or row.y_type == y_type:
            rows.append(row)
    return rows


def CollectRealDesign(test_dir: Path, y_type: str) -> list[DesignSummary]:
    rows = []
    for path in sorted(test_dir.glob('*.npz')):
        with np.load(path, allow_pickle=True) as data:
            if y_type != 'all' and _realYType(data) != y_type:
                continue
            X = np.column_stack([np.ones(_scalar(data, 'n')), np.asarray(data['X'])])
            rows.append(
                _designSummary(
                    source='real',
                    name=path.stem,
                    X=X,
                    groups=np.asarray(data['groups']),
                    d=_scalar(data, 'd'),
                    q=None,
                )
            )
    return rows


def _generatedPaths(data_id: str, glob_patterns: list[str], max_files: int) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in glob_patterns:
        for p in sorted((DATA_DIR / data_id).glob(pattern)):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    paths.sort()
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


def CollectGeneratedDesign(
    data_ids: Iterable[str],
    glob_pattern: str,
    max_files: int,
    max_rows_per_file: int,
    seed: int,
) -> list[DesignSummary]:
    rows = []
    rng = np.random.default_rng(seed)
    for data_id in data_ids:
        for path in _generatedPaths(data_id, glob_pattern, max_files):
            with np.load(path, allow_pickle=True) as data:
                n_batch = len(data['n'])
                if max_rows_per_file > 0 and n_batch > max_rows_per_file:
                    indices = np.sort(rng.choice(n_batch, size=max_rows_per_file, replace=False))
                else:
                    indices = np.arange(n_batch)

                X_all = data['X']
                groups_all = data['groups']
                for i in indices:
                    n = int(data['n'][i])
                    d = int(data['d'][i])
                    q = int(data['q'][i])
                    rows.append(
                        _designSummary(
                            source=data_id,
                            name=f'{data_id}:{path.stem}:{i}',
                            X=X_all[i, :n, :d],
                            groups=groups_all[i, :n],
                            d=d,
                            q=q,
                        )
                    )
            gc.collect()
    return rows


def CollectGeneratedParams(
    data_ids: Iterable[str], glob_pattern: str, max_files: int
) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {
        'sigma_rfx': [],
        'sigma_eps': [],
        'abs_corr_rfx': [],
        'r_squared': [],
        'eta_rfx': [],
        'between_df': [],
        'within_df_min': [],
    }

    for data_id in data_ids:
        for path in _generatedPaths(data_id, glob_pattern, max_files):
            with np.load(path, allow_pickle=True) as data:
                q_arr = data['q'].astype(int)
                m_arr = data['m'].astype(int)
                ns_arr = data['ns']

                for i, q in enumerate(q_arr):
                    m = m_arr[i]
                    out['sigma_rfx'].extend(np.asarray(data['sigma_rfx'][i, :q], dtype=float))
                    out['between_df'].append(float(m - q * (q + 1) // 2))
                    out['within_df_min'].append(float(int(ns_arr[i, :m].min()) - q))
                    if q > 1:
                        corr = np.asarray(data['corr_rfx'][i, :q, :q], dtype=float)
                        tri = np.tril_indices(q, k=-1)
                        out['abs_corr_rfx'].extend(np.abs(corr[tri]))

                for key in ('sigma_eps', 'r_squared', 'eta_rfx'):
                    if key in data.files:
                        out[key].extend(np.asarray(data[key], dtype=float).ravel())

    return {key: np.asarray(values, dtype=float) for key, values in out.items()}


def _quantiles(values: np.ndarray) -> str:
    qs = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    return ' '.join(f'{q:>8.2f}' for q in qs)


def _safeCorr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


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


def PrintDesignSummary(label: str, rows: list[DesignSummary]) -> None:
    if not rows:
        print(f'\n{label}: no design rows')
        return

    print(f'\n--- {label} design diagnostics ({len(rows)} datasets) ---')
    print(f"{'metric':<18} {'min':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'max':>8}")
    for name, values in [
        ('X_rank_gap', np.array([r.x_rank_gap for r in rows], dtype=float)),
        ('X_condition', np.array([r.x_condition for r in rows], dtype=float)),
        (
            'min_Z_rank',
            np.array([r.min_group_z_rank for r in rows if r.min_group_z_rank is not None]),
        ),
        (
            'min_Z_df',
            np.array([r.min_group_z_df for r in rows if r.min_group_z_df is not None]),
        ),
    ]:
        values = _finite(values.astype(float, copy=False))
        if len(values):
            print(f'{name:<18} {_quantiles(values)}')

    rank_deficient = [r for r in rows if r.x_rank_gap > 0]
    ill_conditioned = [r for r in rows if np.isfinite(r.x_condition) and r.x_condition > 1e4]
    singular = [r for r in rows if not np.isfinite(r.x_condition)]
    print(f'X rank deficient          {len(rank_deficient):>8}/{len(rows)}')
    print(f'X condition > 1e4         {len(ill_conditioned):>8}/{len(rows)}')
    print(f'X condition non-finite    {len(singular):>8}/{len(rows)}')
    if rank_deficient or ill_conditioned or singular:
        flagged = rank_deficient + ill_conditioned + singular
        seen = set()
        print('flagged examples:')
        for r in flagged:
            if r.name in seen:
                continue
            seen.add(r.name)
            print(
                f'  {r.name}: d={r.d}, m={r.m}, n={r.n}, '
                f'rank_gap={r.x_rank_gap}, cond={r.x_condition:.2e}'
            )
            if len(seen) >= 8:
                break


def PrintGeneratedParameterSummary(params: dict[str, np.ndarray]) -> None:
    print('\n--- generated parameter and df coverage ---')
    print(f"{'metric':<18} {'min':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'max':>8}")
    for name, values in params.items():
        values = _finite(values)
        if len(values):
            print(f'{name:<18} {_quantiles(values)}')

    between = params.get('between_df', np.array([]))
    within = params.get('within_df_min', np.array([]))
    if len(between):
        print(f'between_df < 4           {int(np.sum(between < 4)):>8}/{len(between)}')
    if len(within):
        print(f'within_df_min < 2        {int(np.sum(within < 2)):>8}/{len(within)}')


def PrintRealOutcomeSummary(test_dir: Path) -> None:
    rows = []
    for path in sorted(test_dir.glob('*.npz')):
        with np.load(path, allow_pickle=True) as data:
            y_type = _realYType(data)
            y = np.asarray(data['y'])
            row = {
                'name': path.stem,
                'y_type': y_type,
                'n': len(y),
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
            }
            if y_type == 'binary':
                row['event_rate'] = float(np.mean(y == 1))
            elif y_type == 'count':
                row['zero_rate'] = float(np.mean(y == 0))
                row['var_over_mean'] = float(np.var(y) / max(np.mean(y), 1e-12))
            rows.append(row)

    print('\n--- real test outcome-shape checks ---')
    counts = {
        y_type: sum(r['y_type'] == y_type for r in rows)
        for y_type in sorted({r['y_type'] for r in rows})
    }
    print('y_type counts:', ', '.join(f'{key}={value}' for key, value in counts.items()))

    binary = [r for r in rows if r['y_type'] == 'binary']
    count = [r for r in rows if r['y_type'] == 'count']
    if binary:
        rates = np.array([r['event_rate'] for r in binary], dtype=float)
        print(f'binary event_rate         {_quantiles(rates)}')
    if count:
        zero = np.array([r['zero_rate'] for r in count], dtype=float)
        over = np.array([r['var_over_mean'] for r in count], dtype=float)
        print(f'count zero_rate           {_quantiles(zero)}')
        print(f'count var/mean            {_quantiles(over)}')


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
    real_design_rows = CollectRealDesign(args.test_dir, y_type=args.y_type)
    generated_rows = CollectGenerated(
        args.generated_data_ids,
        glob_pattern=args.generated_glob,
        max_files=args.max_generated_files,
    )
    generated_design_rows = CollectGeneratedDesign(
        args.generated_data_ids,
        glob_pattern=args.generated_glob,
        max_files=args.max_generated_design_files,
        max_rows_per_file=args.max_generated_design_rows,
        seed=args.design_seed,
    )
    generated_params = CollectGeneratedParams(
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
    PrintDesignSummary('real test pool', real_design_rows)
    PrintDesignSummary('generated training sample', generated_design_rows)
    PrintGeneratedParameterSummary(generated_params)
    PrintRealOutcomeSummary(args.test_dir)
    PrintPriorSummary(real_rows)

    if not args.no_plots:
        Plot(real_rows, generated_rows, args.outdir)


if __name__ == '__main__':
    main()
