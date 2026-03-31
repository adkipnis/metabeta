"""
Runtime comparison: metabeta (with pseudo-MoE) vs NUTS vs ADVI.

For a given evaluation config (= trained model):
    1. Load the model
    2. Discover ALL .fit.npz test files with matching likelihood family
    3. For each file, load individual datasets and pad them to the model's
       maximal dimensions (smaller d/q datasets are zero-padded)
    4. Run model inference using pseudo-MoE to match the NUTS sample count
    5. Collect per-dataset runtimes for model, NUTS, and ADVI
    6. Produce a summary table grouped by source data config

Usage (from experiments/):
    uv run python runtimes.py
    uv run python runtimes.py --configs large-n-sampled
    uv run python runtimes.py --configs small-n-sampled large-n-sampled --k 7
    uv run python runtimes.py --configs large-n-sampled --max_datasets 32
"""

import argparse
import time
import yaml
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate
from tqdm import tqdm

from metabeta.models.approximator import Approximator
from metabeta.utils.config import (
    assimilateConfig,
    loadDataConfig,
    modelFromYaml,
)
from metabeta.utils.dataloader import collateGrouped, toDevice
from metabeta.utils.io import runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.moe import moeEstimate
from metabeta.utils.padding import padToModel, unpad
from metabeta.utils.plot import niceify
from metabeta.utils.sampling import setSeed

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
DATA_DIR = METABETA / 'outputs' / 'data'
OUT_DIR = DIR / 'results'

LIKELIHOOD_NAMES = {0: 'normal', 1: 'bernoulli', 2: 'poisson'}

DEFAULT_CONFIGS = ['small-n-mixed', 'mid-n-mixed', 'medium-n-mixed', 'big-n-mixed', 'large-n-mixed']


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Runtime comparison: metabeta vs NUTS vs ADVI.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names (YAML files in evaluation/configs/)')
    parser.add_argument('--k', type=int, default=0, help='number of extra MoE permuted views (0 = no MoE)')
    parser.add_argument('--max_datasets', type=int, default=None, help='max datasets per fit file (for quick testing)')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR), help='output directory for tables')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    parser.add_argument('--cuda', action='store_true', help='override eval config device and use CUDA (errors if unavailable)')
    return parser.parse_args()
# fmt: on


def loadEvalConfig(name: str, **overrides) -> argparse.Namespace:
    """Load an evaluation YAML config and apply overrides."""
    path = EVAL_CFG_DIR / f'{name}.yaml'
    assert path.exists(), f'eval config not found: {path}'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['name'] = name
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def initModel(cfg: argparse.Namespace, device: torch.device):
    """Load model architecture from config and restore checkpoint weights."""
    data_cfg = loadDataConfig(cfg.d_tag)
    assimilateConfig(cfg, data_cfg)

    model_cfg_path = METABETA / 'models' / 'configs' / f'{cfg.m_tag}.yaml'
    model_cfg = modelFromYaml(
        model_cfg_path,
        d_ffx=cfg.max_d,
        d_rfx=cfg.max_q,
        likelihood_family=getattr(cfg, 'likelihood_family', 0),
    )
    model = Approximator(model_cfg).to(device)
    model.eval()

    run = runName(vars(cfg))
    ckpt_dir = METABETA / 'outputs' / 'checkpoints' / run
    path = ckpt_dir / f'{cfg.prefix}.pt'
    assert path.exists(), f'checkpoint not found: {path}'
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload['model_state'])

    if cfg.compile and device.type != 'mps':
        model.compile()

    return model, data_cfg


def discoverFitFiles(likelihood_family: int) -> list[Path]:
    """Find all .fit.npz test files matching the given likelihood family."""
    lf_name = LIKELIHOOD_NAMES[likelihood_family]
    pattern = f'test_{lf_name}_*.fit.npz'
    paths = sorted(DATA_DIR.glob(pattern))
    assert len(paths) > 0, f'no fit files found for pattern {pattern} in {DATA_DIR}'
    return paths


def extractDatasets(
    path: Path,
    max_d: int,
    max_q: int,
    max_datasets: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Load a fit.npz file, unpad individual datasets, and re-pad to model dims."""
    with np.load(path, allow_pickle=True) as raw:
        raw = dict(raw)

    B = len(raw['d'])
    n_use = min(B, max_datasets) if max_datasets else B
    datasets = []

    for i in range(n_use):
        ds = {k: v[i] for k, v in raw.items()}
        # skip datasets whose d or q exceeds the model's capacity
        if int(ds['d']) > max_d or int(ds['q']) > max_q:
            continue
        # unpad to actual dimensions
        sizes = {k: int(ds[k]) for k in ('d', 'q', 'm', 'n')}
        ds = unpad(ds, sizes)
        # re-pad to model dimensions
        ds = padToModel(ds, max_d, max_q)
        datasets.append(ds)

    return datasets


def resetRng(model: Approximator, seed: int) -> None:
    """Reset base distribution RNGs for reproducible sampling."""
    model.posterior_g.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore
    model.posterior_l.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore


@torch.inference_mode()
def benchmarkModel(
    model: Approximator,
    datasets: list[dict[str, np.ndarray]],
    n_samples: int,
    k: int,
    device: torch.device,
    seed: int,
    rescale: bool,
) -> np.ndarray:
    """Run model inference on each dataset and return per-dataset runtimes."""
    durations = np.zeros(len(datasets))

    # warmup pass to avoid cold-start overhead in the first timed call
    warmup_batch = collateGrouped([datasets[0]])
    warmup_batch = toDevice(warmup_batch, device)
    resetRng(model, seed)
    moeEstimate(model, warmup_batch, n_samples, k, rng=np.random.default_rng(0))

    for i, ds in enumerate(tqdm(datasets, desc='  metabeta')):
        batch = collateGrouped([ds])
        batch = toDevice(batch, device)

        setSeed(seed)
        resetRng(model, seed)
        rng = np.random.default_rng(seed + i)

        t0 = time.perf_counter()
        proposal = moeEstimate(model, batch, n_samples, k, rng=rng)
        if rescale:
            proposal.rescale(batch['sd_y'])
        t1 = time.perf_counter()

        durations[i] = t1 - t0

    return durations


def sourceLabel(path: Path) -> str:
    """Extract a short label from a fit file name, e.g. 'd3_q1_m5-25'."""
    stem = path.stem.replace('.fit', '')
    # test_normal_d3_q1_m5-25_n4-18_sampled_nt800_all → d3_q1_m5-25_n4-18
    parts = stem.split('_')
    # find d/q/m/n parts
    dqmn = [p for p in parts if p[0] in 'dqmn' and p[1:2].isdigit()]
    return '_'.join(dqmn)


def nParams(ds: dict[str, np.ndarray]) -> int:
    """Effective number of model parameters: d (ffx) + q (sigma) + m*q (rfx)."""
    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    return d + q + m * q


def evaluateConfig(
    config_name: str,
    k: int,
    max_datasets: int | None,
    cuda: bool = False,
) -> list[dict]:
    """Run the runtime comparison for a single config, returning per-dataset records."""
    cfg = loadEvalConfig(config_name, plot=False)
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('--cuda specified but CUDA is not available on this machine')
        cfg.device = 'cuda'
    setSeed(cfg.seed)
    device = setDevice(cfg.device)

    model, data_cfg = initModel(cfg, device)
    max_d = cfg.max_d
    max_q = cfg.max_q
    lf = getattr(cfg, 'likelihood_family', data_cfg.get('likelihood_family', 0))

    print(f'Model: {config_name} (max_d={max_d}, max_q={max_q}, lf={lf})')

    fit_paths = discoverFitFiles(lf)
    print(f'Found {len(fit_paths)} fit file(s)')

    records = []

    for path in fit_paths:
        label = sourceLabel(path)
        print(f'\n{"=" * 60}')
        print(f'Source: {path.name} ({label})')

        datasets = extractDatasets(path, max_d, max_q, max_datasets)
        if not datasets:
            print('  (skipped — no compatible datasets)')
            continue

        N = len(datasets)
        ds_d = np.array([int(ds['d']) for ds in datasets])
        ds_q = np.array([int(ds['q']) for ds in datasets])

        # NUTS sample count from first dataset
        nuts_n_samples = datasets[0]['nuts_ffx'].shape[-1]
        n_samples_per_view = nuts_n_samples // (1 + k)
        total_samples = n_samples_per_view * (1 + k)

        print(f'  {N} datasets, d={ds_d.min()}-{ds_d.max()}, q={ds_q.min()}-{ds_q.max()}')
        print(f'  NUTS draws={nuts_n_samples}, model: {n_samples_per_view}×{1+k}={total_samples}')

        # model runtimes
        model_dur = benchmarkModel(
            model, datasets, n_samples_per_view, k, device, cfg.seed, cfg.rescale
        )

        # NUTS and ADVI runtimes (pre-computed in fit file)
        with np.load(path, allow_pickle=True) as raw:
            nuts_dur = raw['nuts_duration'][:N].astype(np.float64)
            advi_dur = raw['advi_duration'][:N].astype(np.float64)

        for i, ds in enumerate(datasets):
            base = {
                'config': config_name,
                'source': label,
                'n': int(ds['n']),
                'd': int(ds['d']),
                'q': int(ds['q']),
                'm': int(ds['m']),
                'n_params': nParams(ds),
            }
            for method, dur in [('metabeta', model_dur), ('NUTS', nuts_dur), ('ADVI', advi_dur)]:
                records.append({**base, 'method': method, 'duration': dur[i]})

    return records


def evaluate(
    configs: list[str],
    k: int,
    max_datasets: int | None,
    cuda: bool = False,
) -> list[dict]:
    """Run the full runtime comparison across all configs."""
    records = []
    for config_name in configs:
        print(f'\n{"#" * 60}')
        print(f'Config: {config_name}')
        print(f'{"#" * 60}')
        records.extend(evaluateConfig(config_name, k, max_datasets, cuda=cuda))
    return records


def _groupKey(r: dict) -> tuple:
    return (r['config'], r['source'], r['method'])


def _aggregateRows(records: list[dict]) -> list[dict]:
    """Aggregate per-dataset records into per-(config, source, method) summary rows."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[_groupKey(r)].append(r)

    rows = []
    for (cfg, src, method), recs in groups.items():
        durs = np.array([r['duration'] for r in recs])
        ds_d = np.array([r['d'] for r in recs])
        ds_q = np.array([r['q'] for r in recs])
        ds_m = np.array([r['m'] for r in recs])
        ds_n = np.array([r['n'] for r in recs])
        rows.append(
            {
                'config': cfg,
                'source': src,
                'method': method,
                'N': len(recs),
                'd': f'{ds_d.min()}-{ds_d.max()}',
                'q': f'{ds_q.min()}-{ds_q.max()}',
                'm': f'{ds_m.min()}-{ds_m.max()}',
                'n': f'{ds_n.min()}-{ds_n.max()}',
                'median_t': np.median(durs),
                'mean_t': np.mean(durs),
                'total_t': np.sum(durs),
            }
        )
    return rows


def formatTable(records: list[dict]) -> str:
    """Format absolute runtimes table."""
    rows = _aggregateRows(records)
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r['config'],
                r['source'],
                r['method'],
                r['N'],
                r['d'],
                r['q'],
                r['m'],
                r['n'],
                f'{r["median_t"]:.3f}',
                f'{r["mean_t"]:.3f}',
                f'{r["total_t"]:.1f}',
            ]
        )
    headers = [
        'Config',
        'Source',
        'Method',
        'N',
        'd',
        'q',
        'm',
        'n',
        'Median [s]',
        'Mean [s]',
        'Total [s]',
    ]
    return tabulate(table_rows, headers=headers, tablefmt='pipe', stralign='right')


def speedupTable(records: list[dict]) -> str:
    """Format speedup table (relative to NUTS)."""
    rows = _aggregateRows(records)

    # group by (config, source)
    keys = []
    seen = set()
    for r in rows:
        key = (r['config'], r['source'])
        if key not in seen:
            keys.append(key)
            seen.add(key)

    table_rows = []
    for cfg, src in keys:
        src_rows = {r['method']: r for r in rows if r['config'] == cfg and r['source'] == src}
        if 'NUTS' not in src_rows or 'metabeta' not in src_rows:
            continue
        nuts_med = src_rows['NUTS']['median_t']
        model_med = src_rows['metabeta']['median_t']
        advi_med = src_rows.get('ADVI', {}).get('median_t')

        table_rows.append(
            [
                cfg,
                src,
                src_rows['metabeta']['d'],
                src_rows['metabeta']['q'],
                src_rows['metabeta']['m'],
                src_rows['metabeta']['n'],
                f'{nuts_med:.3f}',
                f'{model_med:.3f}',
                f'{nuts_med / model_med:.0f}x' if model_med > 0 else '-',
                f'{advi_med:.3f}' if advi_med else '-',
                f'{advi_med / model_med:.0f}x' if advi_med and model_med > 0 else '-',
            ]
        )

    headers = [
        'Config',
        'Source',
        'd',
        'q',
        'm',
        'n',
        'NUTS [s]',
        'Model [s]',
        'Speedup',
        'ADVI [s]',
        'vs ADVI',
    ]
    return tabulate(table_rows, headers=headers, tablefmt='pipe', stralign='right')


METHOD_COLORS = {'metabeta': '#1f77b4', 'NUTS': '#d62728', 'ADVI': '#ff7f0e'}
METHOD_MARKERS = {'metabeta': 'o', 'NUTS': 's', 'ADVI': 'D'}


def plotRuntimes(records: list[dict], outdir: Path) -> Path:
    """Scatter plot: total n vs runtime, size ~ n_params, color ~ method.

    Each point is one dataset.  Both axes are log-scaled.  Marker size is
    proportional to the effective number of parameters (d + q + m*q), giving
    an immediate sense of model complexity.  Thin vertical lines connect the
    three methods for the same dataset so the speedup gap is easy to read.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # size scaling: map n_params to marker area
    all_np = np.array([r['n_params'] for r in records])
    np_min, np_max = all_np.min(), all_np.max()
    s_min, s_max = 25, 220
    if np_max > np_min:
        scale = lambda p: s_min + (s_max - s_min) * (p - np_min) / (np_max - np_min)
    else:
        scale = lambda p: (s_min + s_max) / 2

    # draw connector lines between methods for each dataset
    from collections import defaultdict

    dataset_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r['config'], r['source'], r['d'], r['q'], r['m'], r['n'])
        dataset_groups[key].append(r)

    for key, group in dataset_groups.items():
        if len(group) < 2:
            continue
        n = key[-1]  # total observations
        durs = [r['duration'] for r in group]
        ax.plot(
            [n, n],
            [min(durs), max(durs)],
            color='#cccccc',
            linewidth=0.7,
            zorder=1,
        )

    # scatter by method
    for method in ['NUTS', 'ADVI', 'metabeta']:
        sub = [r for r in records if r['method'] == method]
        if not sub:
            continue
        x = np.array([r['n'] for r in sub])
        y = np.array([r['duration'] for r in sub])
        s = np.array([scale(r['n_params']) for r in sub])
        ax.scatter(
            x,
            y,
            s=s,
            c=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            alpha=0.7,
            edgecolors='white',
            linewidths=0.4,
            label=method,
            zorder=3,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, which='both')

    # method legend
    niceify(
        ax,
        {
            'title': 'Runtime Comparison',
            'xlabel': 'Total observations (n)',
            'ylabel': 'Time [s]',
            'title_fs': 20,
            'xlabel_fs': 16,
            'ylabel_fs': 16,
            'ticks_ls': 13,
            'legend_fs': 13,
            'legend_ms': 1.5,
            'legend_loc': 'upper left',
            'despine': True,
        },
    )

    # size legend (n_params)
    n_unique = len(np.unique(all_np))
    size_ticks = np.linspace(np_min, np_max, min(4, n_unique))
    size_ticks = np.unique(np.round(size_ticks).astype(int))
    size_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            markerfacecolor='grey',
            markeredgecolor='none',
            markersize=np.sqrt(scale(v)),
            label=str(v),
        )
        for v in size_ticks
    ]
    size_leg = ax.legend(
        handles=size_handles,
        title='n params',
        fontsize=11,
        title_fontsize=12,
        loc='lower right',
        framealpha=0.8,
    )
    ax.add_artist(size_leg)
    # restore method legend
    ax.legend(fontsize=13, markerscale=1.5, loc='upper left')

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'runtimes.png'
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(outdir / 'runtimes.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    return path


def save(records: list[dict], outdir: Path) -> None:
    """Save tables and plot."""
    outdir.mkdir(parents=True, exist_ok=True)

    abs_table = formatTable(records)
    spd_table = speedupTable(records)

    md_path = outdir / 'runtimes.md'
    md_path.write_text(
        f'# Runtime Comparison\n\n'
        f'## Absolute Runtimes\n\n{abs_table}\n\n'
        f'## Speedup vs NUTS\n\n{spd_table}\n'
    )
    print(f'\nMarkdown saved to {md_path}')

    fig_path = plotRuntimes(records, outdir)
    print(f'Plot saved to {fig_path}')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)

    print(f'Runtime comparison: {len(args.configs)} config(s)')
    print(f'Configs: {args.configs}')
    print(f'MoE k={args.k}')

    records = evaluate(args.configs, args.k, args.max_datasets, cuda=args.cuda)

    print(f'\n{formatTable(records)}')
    print(f'\n{speedupTable(records)}')

    save(records, Path(args.outdir))
    print('\nDone.')
