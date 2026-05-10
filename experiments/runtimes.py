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
    uv run python runtimes.py --configs small-n-mixed --model_id large --seed 0
    uv run python runtimes.py --configs small-n-mixed --test_data_ids tiny-n-sampled small-n-sampled
    uv run python runtimes.py --configs small-n-mixed --max_test_sets 2 --max_datasets 32
"""

import argparse
import json
import time
import yaml
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.models.approximator import Approximator
from metabeta.plotting.runtimes import plotRuntimeRecords
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
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='training data config names, or YAML files in evaluation/configs/')
    parser.add_argument('--model_id', default='large', help='model config id used when --configs names data ids')
    parser.add_argument('--seed', type=int, default=0, help='checkpoint seed used when --configs names data ids')
    parser.add_argument('--prefix', default='best', help='checkpoint prefix to load')
    parser.add_argument('--k', type=int, default=0, help='number of extra MoE permuted views (0 = no MoE)')
    parser.add_argument('--test_data_ids', nargs='+', default=None, help='specific outputs/data/* test sets to benchmark')
    parser.add_argument('--max_test_sets', type=int, default=None, help='max number of test.fit.npz files to benchmark')
    parser.add_argument('--max_datasets', type=int, default=None, help='max datasets per fit file (for quick testing)')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR), help='output directory for tables')
    parser.add_argument('--cache_path', type=Path, default=None, help='JSON cache for metabeta runtimes (default: outdir/runtimes_cache.json)')
    parser.add_argument('--refresh_cache', action='store_true', help='recompute metabeta runtimes even if cached')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    parser.add_argument('--cuda', action='store_true', help='override eval config device and use CUDA (errors if unavailable)')
    return parser.parse_args()
# fmt: on


def loadEvalConfig(
    name: str,
    model_id: str = 'large',
    seed: int = 0,
    prefix: str = 'best',
    **overrides,
) -> argparse.Namespace:
    """Load an evaluation YAML or checkpoint config and apply overrides."""
    path = EVAL_CFG_DIR / f'{name}.yaml'
    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            'data_id': name,
            'model_id': model_id,
            'seed': seed,
            'r_tag': '',
            'likelihood_family': 0,
        }
        ckpt_cfg = METABETA / 'outputs' / 'checkpoints' / runName(cfg) / 'config.yaml'
        assert ckpt_cfg.exists(), f'eval config not found: {path} or {ckpt_cfg}'
        with open(ckpt_cfg) as f:
            cfg = yaml.safe_load(f)
    cfg['name'] = name
    cfg.setdefault('device', 'cpu')
    cfg.setdefault('compile', False)
    cfg.setdefault('prefix', prefix)
    cfg.setdefault('r_tag', '')
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def initModel(cfg: argparse.Namespace, device: torch.device):
    """Load model architecture from config and restore checkpoint weights."""
    data_cfg = loadDataConfig(cfg.data_id)
    assimilateConfig(cfg, data_cfg)

    model_cfg_path = METABETA / 'configs' / 'models' / f'{cfg.model_id}.yaml'
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
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'])

    if cfg.compile and device.type != 'mps':
        model.compile()

    return model, data_cfg


def _fitFileLikelihood(path: Path) -> int | None:
    try:
        return int(loadDataConfig(path.parent.name).get('likelihood_family', 0))
    except FileNotFoundError:
        return None


def discoverFitFiles(
    likelihood_family: int,
    test_data_ids: list[str] | None = None,
    max_test_sets: int | None = None,
) -> list[Path]:
    """Find all .fit.npz test files matching the given likelihood family."""
    lf_name = LIKELIHOOD_NAMES[likelihood_family]
    if test_data_ids is not None:
        paths = [DATA_DIR / data_id / 'test.fit.npz' for data_id in test_data_ids]
    else:
        pattern = f'test_{lf_name}_*.fit.npz'
        paths = sorted(DATA_DIR.glob(pattern))
        paths.extend(
            p
            for p in sorted(DATA_DIR.glob('*/test.fit.npz'))
            if _fitFileLikelihood(p) == likelihood_family
        )

    paths = [p for p in paths if p.exists()]
    if max_test_sets is not None:
        paths = paths[:max_test_sets]
    assert len(paths) > 0, f'no fit files found for likelihood {lf_name} in {DATA_DIR}'
    return paths


def extractDatasets(
    path: Path,
    max_d: int,
    max_q: int,
    max_datasets: int | None = None,
) -> list[tuple[int, dict[str, np.ndarray]]]:
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
        datasets.append((i, ds))

    return datasets


def resetRng(model: Approximator, seed: int) -> None:
    """Reset base distribution RNGs for reproducible sampling."""
    base_g = model.posterior_g.base_dist
    if hasattr(base_g, 'base') and hasattr(base_g.base, 'rng'):
        base_g.base.rng = np.random.default_rng(seed)  # type: ignore
    if hasattr(model, 'posterior_l'):
        base_l = model.posterior_l.base_dist
        if hasattr(base_l, 'base') and hasattr(base_l.base, 'rng'):
            base_l.base.rng = np.random.default_rng(seed)  # type: ignore


@torch.inference_mode()
def benchmarkModel(
    model: Approximator,
    datasets: list[dict[str, np.ndarray]],
    dataset_idxs: list[int],
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
        rng = np.random.default_rng(seed + dataset_idxs[i])

        t0 = time.perf_counter()
        proposal = moeEstimate(model, batch, n_samples, k, rng=rng)
        if rescale:
            proposal.rescale(batch['sd_y'])
        t1 = time.perf_counter()

        durations[i] = t1 - t0

    return durations


def sourceLabel(path: Path) -> str:
    """Extract a short label from a fit file name, e.g. 'd3_q1_m5-25'."""
    if path.name == 'test.fit.npz':
        return path.parent.name
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


def _cacheKey(
    cfg: argparse.Namespace,
    source: str,
    idx: int,
    n_samples: int,
    k: int,
    device: torch.device,
) -> str:
    parts = [
        str(cfg.data_id),
        str(cfg.model_id),
        str(cfg.seed),
        str(cfg.prefix),
        str(source),
        str(idx),
        str(n_samples),
        str(k),
        str(device.type),
    ]
    return '|'.join(parts)


def loadRuntimeCache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'durations' in raw:
        raw = raw['durations']
    return {str(k): float(v) for k, v in raw.items()}


def saveRuntimeCache(path: Path, cache: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    payload = {'durations': dict(sorted(cache.items()))}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def evaluateConfig(
    config_name: str,
    model_id: str,
    seed: int,
    prefix: str,
    k: int,
    test_data_ids: list[str] | None,
    max_test_sets: int | None,
    max_datasets: int | None,
    runtime_cache: dict[str, float],
    cache_path: Path,
    refresh_cache: bool = False,
    cuda: bool = False,
) -> list[dict]:
    """Run the runtime comparison for a single config, returning per-dataset records."""
    cfg = loadEvalConfig(config_name, model_id=model_id, seed=seed, prefix=prefix, plot=False)
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

    fit_paths = discoverFitFiles(lf, test_data_ids=test_data_ids, max_test_sets=max_test_sets)
    print(f'Found {len(fit_paths)} fit file(s)')

    records = []

    for path in fit_paths:
        label = sourceLabel(path)
        print(f'\n{"=" * 60}')
        print(f'Source: {path.name} ({label})')

        indexed_datasets = extractDatasets(path, max_d, max_q, max_datasets)
        if not indexed_datasets:
            print('  (skipped — no compatible datasets)')
            continue
        raw_idxs = [idx for idx, _ in indexed_datasets]
        datasets = [ds for _, ds in indexed_datasets]

        N = len(datasets)
        ds_d = np.array([int(ds['d']) for ds in datasets])
        ds_q = np.array([int(ds['q']) for ds in datasets])

        # NUTS sample count from first dataset
        nuts_n_samples = datasets[0]['nuts_ffx'].shape[-1]
        n_samples_per_view = nuts_n_samples // (1 + k)
        total_samples = n_samples_per_view * (1 + k)

        print(f'  {N} datasets, d={ds_d.min()}-{ds_d.max()}, q={ds_q.min()}-{ds_q.max()}')
        print(f'  NUTS draws={nuts_n_samples}, model: {n_samples_per_view}×{1+k}={total_samples}')

        cache_keys = [
            _cacheKey(cfg, label, raw_idx, n_samples_per_view, k, device) for raw_idx in raw_idxs
        ]
        model_dur = np.full(N, np.nan)
        missing_pos = []
        for i, key in enumerate(cache_keys):
            if not refresh_cache and key in runtime_cache:
                model_dur[i] = runtime_cache[key]
            else:
                missing_pos.append(i)

        n_cached = N - len(missing_pos)
        if n_cached:
            print(f'  metabeta cache hits={n_cached}/{N}')
        if missing_pos:
            missing_datasets = [datasets[i] for i in missing_pos]
            missing_idxs = [raw_idxs[i] for i in missing_pos]
            missing_dur = benchmarkModel(
                model,
                missing_datasets,
                missing_idxs,
                n_samples_per_view,
                k,
                device,
                cfg.seed,
                cfg.rescale,
            )
            for pos, dur in zip(missing_pos, missing_dur):
                model_dur[pos] = dur
                runtime_cache[cache_keys[pos]] = float(dur)
            saveRuntimeCache(cache_path, runtime_cache)

        # NUTS and ADVI runtimes (pre-computed in fit file)
        with np.load(path, allow_pickle=True) as raw:
            nuts_dur = raw['nuts_duration'][raw_idxs].astype(np.float64)
            advi_dur = raw['advi_duration'][raw_idxs].astype(np.float64)

        for i, (raw_idx, ds) in enumerate(zip(raw_idxs, datasets)):
            base = {
                'config': config_name,
                'source': label,
                'idx': raw_idx,
                'n': int(ds['n']),
                'd': int(ds['d']),
                'q': int(ds['q']),
                'm': int(ds['m']),
                'n_params': nParams(ds),
            }
            for method, dur in [('metabeta', model_dur), ('NUTS', nuts_dur), ('ADVI', advi_dur)]:
                records.append({**base, 'method': method, 'duration': float(dur[i])})

    return records


def evaluate(
    configs: list[str],
    model_id: str,
    seed: int,
    prefix: str,
    k: int,
    test_data_ids: list[str] | None,
    max_test_sets: int | None,
    max_datasets: int | None,
    cache_path: Path,
    refresh_cache: bool = False,
    cuda: bool = False,
) -> list[dict]:
    """Run the full runtime comparison across all configs."""
    runtime_cache = loadRuntimeCache(cache_path)
    records = []
    for config_name in configs:
        print(f'\n{"#" * 60}')
        print(f'Config: {config_name}')
        print(f'{"#" * 60}')
        records.extend(
            evaluateConfig(
                config_name,
                model_id,
                seed,
                prefix,
                k,
                test_data_ids,
                max_test_sets,
                max_datasets,
                runtime_cache,
                cache_path,
                refresh_cache=refresh_cache,
                cuda=cuda,
            )
        )
    return records


def splitRecordsByDevice(
    records: list[dict],
    cache: dict[str, float],
    devices: list[str] | None = None,
) -> list[dict]:
    """Replace 'metabeta' records with per-device variants using the runtime cache.

    Cache keys: config|model_id|seed|prefix|source|idx|n_samples|k|device
    Output method names: 'metabeta_gpu' (cuda) and 'metabeta_cpu' (cpu).
    Non-metabeta records (NUTS, ADVI) are passed through unchanged.
    """
    if devices is None:
        devices = ['cuda', 'cpu']
    device_suffix = {'cuda': 'metabeta_gpu', 'cpu': 'metabeta_cpu'}

    # Infer common metadata from existing cache keys
    sample_key = next(iter(cache))
    parts = sample_key.split('|')
    model_id, seed, prefix, n_samples, k = parts[1], parts[2], parts[3], parts[6], parts[7]

    out = [r for r in records if r['method'] != 'metabeta']
    for r in records:
        if r['method'] != 'metabeta':
            continue
        for device in devices:
            key = '|'.join([r['config'], model_id, seed, prefix, r['source'],
                            str(r['idx']), n_samples, k, device])
            if key not in cache:
                continue
            out.append({**r, 'method': device_suffix[device], 'duration': cache[key]})
    return out


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

    records_path = outdir / 'runtimes_records.json'
    records_path.write_text(json.dumps(records, indent=2, sort_keys=True) + '\n')
    print(f'Records saved to {records_path}')

    fig_path = plotRuntimeRecords(records, out_dir=outdir)
    print(f'Plot saved to {fig_path}')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)

    print(f'Runtime comparison: {len(args.configs)} config(s)')
    print(f'Configs: {args.configs}')
    print(f'MoE k={args.k}')

    outdir = Path(args.outdir)
    cache_path = args.cache_path if args.cache_path is not None else outdir / 'runtimes_cache.json'

    records = evaluate(
        args.configs,
        args.model_id,
        args.seed,
        args.prefix,
        args.k,
        args.test_data_ids,
        args.max_test_sets,
        args.max_datasets,
        cache_path,
        refresh_cache=args.refresh_cache,
        cuda=args.cuda,
    )

    print(f'\n{formatTable(records)}')
    print(f'\n{speedupTable(records)}')

    save(records, outdir)
    print('\nDone.')
