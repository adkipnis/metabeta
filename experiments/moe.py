"""
Mixture of Experts experiment: pseudo (permutation) and true (multi-checkpoint) modes.

Pseudo-MoE (default): single checkpoint, k random feature permutations per dataset.
True MoE (--multi)  : one checkpoint per config, proposals mixed across all models.

Usage (from experiments/):
    uv run python moe.py                                              # pseudo, toy-n
    uv run python moe.py --configs small-n-sampled --ks 0 3 7        # pseudo
    uv run python moe.py --multi --configs small-n-sampled small-n-mixed  # true MoE
    uv run python moe.py --multi --configs small-n-sampled small-n-mixed --eval-config small-n-sampled
    uv run python moe.py --valid --importance
"""

import argparse
import time
import yaml
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.models.approximator import Approximator
from metabeta.posthoc.conformal import Calibrator
from metabeta.utils.config import (
    assimilateConfig,
    loadDataConfig,
    modelFromYaml,
)
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.evaluation import Proposal, concatProposalsBatch, dictMean
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.io import datasetFilename, runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.moe import moeEstimate, multiCheckpointEstimate
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.evaluation.summary import getSummary

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results'

DEFAULT_CONFIGS = ['toy-n']
DEFAULT_KS = [0, 3, 7]

# (display name, extractor, higher_is_better)
METRICS = [
    ('R', lambda s: dictMean(s.corr), True),
    ('NRMSE', lambda s: dictMean(s.nrmse), False),
    ('ECE', lambda s: dictMean(s.ece), 'abs'),
    ('ppNLL', lambda s: s.mnll, False),
]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MoE experiment: pseudo (permutation) and true (multi-checkpoint).')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names (YAML files in evaluation/configs/)')
    parser.add_argument('--ks', nargs='+', type=int, default=DEFAULT_KS, help='number of extra permuted views (0 = baseline)')
    parser.add_argument('--importance', action='store_true', help='apply importance sampling on the joined proposal')
    parser.add_argument('--valid', action='store_true', help='use validation set instead of test set')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR), help='output directory for tables')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    # multi-checkpoint MoE mode
    parser.add_argument('--multi', action='store_true', help='true MoE: treat --configs as a list of checkpoints to mix')
    parser.add_argument('--eval-config', type=str, default=None, help='config whose data/calibrator is used for evaluation in --multi mode (default: first of --configs)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='training seeds for --multi mode; a single --configs entry is replicated once per seed')
    # head-to-head comparison mode
    parser.add_argument('--compare', action='store_true', help='compare pseudo-MoE (k=1) vs true MoE (2 experts) at matched sample counts')
    parser.add_argument('--mix-configs', nargs='+', default=None, help='second checkpoint per eval config for --compare mode (must match --configs length)')
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


def initModel(cfg: argparse.Namespace, device: torch.device) -> Approximator:
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

    return model, data_cfg, run


def getDataloader(
    data_cfg: dict, partition: str, batch_size: int | None = None
) -> Dataloader:
    """Create a dataloader for the given partition."""
    data_fname = datasetFilename(data_cfg, partition)
    data_path = METABETA / 'outputs' / 'data' / data_fname
    assert data_path.exists(), f'data not found: {data_path}'
    sortish = batch_size is not None
    return Dataloader(data_path, batch_size=batch_size, sortish=sortish)


def resetRng(model: Approximator, seed: int) -> None:
    """Reset base distribution RNGs for reproducible sampling."""
    model.posterior_g.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore
    model.posterior_l.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore


@torch.inference_mode()
def calibrate(
    model: Approximator,
    cfg: argparse.Namespace,
    data_cfg: dict,
    run: str,
    device: torch.device,
) -> Calibrator:
    """Load or compute conformal calibrator from validation set."""
    calibrator = Calibrator()
    ckpt_path = METABETA / 'outputs' / 'checkpoints' / run / 'calibrator.npz'
    if ckpt_path.exists():
        calibrator.load(run)
    else:
        dl_valid = getDataloader(data_cfg, 'valid')
        batch = next(iter(dl_valid))
        batch = toDevice(batch, device)
        proposal = model.estimate(batch, n_samples=cfg.n_samples)
        if cfg.rescale:
            proposal.rescale(batch['sd_y'])
        batch = toDevice(batch, 'cpu')
        if cfg.rescale:
            batch = rescaleData(batch)
        proposal.to('cpu')
        calibrator.calibrate(proposal, batch)
        calibrator.save(run)
    return calibrator


@torch.inference_mode()
def sampleMoe(
    model: Approximator,
    cfg: argparse.Namespace,
    dl: Dataloader,
    device: torch.device,
    k: int,
    seed: int,
) -> Proposal:
    """Run pseudo-MoE inference over a dataloader, one dataset at a time."""
    proposals = []
    n_datasets = 0
    lf = getattr(cfg, 'likelihood_family', 0)
    t0 = time.perf_counter()

    for batch in tqdm(dl, desc=f'  k={k}'):
        batch = toDevice(batch, device)
        B = batch['X'].shape[0]

        # process each dataset individually (B=1 required by moe)
        for i in range(B):
            single = {
                k_: v[i : i + 1] if torch.is_tensor(v) else v for k_, v in batch.items()
            }
            rng = np.random.default_rng(seed + n_datasets)
            proposal = moeEstimate(model, single, cfg.n_samples, k, rng=rng)
            if cfg.rescale:
                proposal.rescale(single['sd_y'])
            if cfg.importance:
                data_is = rescaleData(single) if cfg.rescale else single
                imp_sampler = ImportanceSampler(data_is, sir=False, likelihood_family=lf)
                proposal = imp_sampler(proposal)
            proposal.to('cpu')
            proposals.append(proposal)
            n_datasets += 1

    t1 = time.perf_counter()
    merged = concatProposalsBatch(proposals)
    merged.tpd = (t1 - t0) / max(n_datasets, 1)
    return merged


@torch.inference_mode()
def sampleMultiMoe(
    models: list[Approximator],
    cfg: argparse.Namespace,
    dl: Dataloader,
    device: torch.device,
    k: int,
    seed: int,
) -> Proposal:
    """Run multi-checkpoint MoE inference over a dataloader, one dataset at a time."""
    proposals = []
    n_datasets = 0
    lf = getattr(cfg, 'likelihood_family', 0)
    t0 = time.perf_counter()

    for batch in tqdm(dl, desc=f'  multi k={k}'):
        batch = toDevice(batch, device)
        B = batch['X'].shape[0]

        for i in range(B):
            single = {
                k_: v[i : i + 1] if torch.is_tensor(v) else v for k_, v in batch.items()
            }
            rng = np.random.default_rng(seed + n_datasets)
            proposal = multiCheckpointEstimate(models, single, cfg.n_samples, k=k, rng=rng)
            if cfg.rescale:
                proposal.rescale(single['sd_y'])
            if cfg.importance:
                data_is = rescaleData(single) if cfg.rescale else single
                imp_sampler = ImportanceSampler(data_is, sir=False, likelihood_family=lf)
                proposal = imp_sampler(proposal)
            proposal.to('cpu')
            proposals.append(proposal)
            n_datasets += 1

    t1 = time.perf_counter()
    merged = concatProposalsBatch(proposals)
    merged.tpd = (t1 - t0) / max(n_datasets, 1)
    return merged


def evaluateComparison(
    eval_configs: list[str],
    mix_configs: list[str],
    use_valid: bool,
    importance: bool,
) -> list[dict]:
    """Head-to-head: pseudo-MoE (k=1) vs true MoE (2 experts, k=0) at equal sample counts.

    For each (eval_config, mix_config) pair, runs three conditions — all on the
    eval config's test data:
      baseline   : eval model, k=0, n_samples       → n_samples total
      pseudo-MoE : eval model, k=1, n_samples       → 2*n_samples total
      true MoE   : [eval_model, mix_model], k=0     → 2*n_samples total
    """
    if len(eval_configs) != len(mix_configs):
        raise ValueError('--configs and --mix-configs must have the same length')

    partition = 'valid' if use_valid else 'test'
    rows = []

    for eval_config, mix_config in zip(eval_configs, mix_configs):
        print(f'\n{"=" * 60}')
        print(f'Comparison: {eval_config} vs +{mix_config}  (partition={partition})')
        print(f'{"=" * 60}')

        eval_cfg = loadEvalConfig(eval_config, plot=False, importance=importance)
        setSeed(eval_cfg.seed)
        device = setDevice(eval_cfg.device)

        eval_model, eval_data_cfg, eval_run = initModel(eval_cfg, device)
        cal = calibrate(eval_model, eval_cfg, eval_data_cfg, eval_run, device)
        calibrator = cal if eval_cfg.conformal else None
        lf = getattr(eval_cfg, 'likelihood_family', 0)

        dl = getDataloader(eval_data_cfg, partition, batch_size=1)
        full_batch = dl.fullBatch()
        if eval_cfg.rescale:
            full_batch = rescaleData(full_batch)

        mix_cfg = loadEvalConfig(mix_config, plot=False, importance=importance)
        if getattr(mix_cfg, 'likelihood_family', 0) != lf:
            raise ValueError(
                f'{mix_config} has different likelihood_family than {eval_config}'
            )
        mix_model, _, _ = initModel(mix_cfg, device)

        n = eval_cfg.n_samples  # per model / per view

        # --- baseline: eval model, k=0 ---
        print(f'\n  --- baseline ({eval_config}, k=0, S={n}) ---')
        setSeed(eval_cfg.seed)
        resetRng(eval_model, eval_cfg.seed)
        eval_model.eval()
        proposal = sampleMoe(eval_model, eval_cfg, dl, device, k=0, seed=eval_cfg.seed)
        summary = getSummary(proposal, full_batch, calibrator=calibrator, likelihood_family=lf)
        rows.append({
            'config': eval_config,
            'condition': 'baseline',
            'total_samples': n,
            **{mn: ex(summary) for mn, ex, _ in METRICS},
            **({'t/ds': summary.tpd} if summary.tpd is not None else {}),
        })

        # --- pseudo-MoE: eval model, k=1 → 2*n samples ---
        print(f'\n  --- pseudo-MoE ({eval_config}, k=1, S={2 * n}) ---')
        setSeed(eval_cfg.seed)
        resetRng(eval_model, eval_cfg.seed)
        eval_model.eval()
        proposal = sampleMoe(eval_model, eval_cfg, dl, device, k=1, seed=eval_cfg.seed)
        summary = getSummary(proposal, full_batch, calibrator=calibrator, likelihood_family=lf)
        rows.append({
            'config': eval_config,
            'condition': 'pseudo-MoE (k=1)',
            'total_samples': 2 * n,
            **{mn: ex(summary) for mn, ex, _ in METRICS},
            **({'t/ds': summary.tpd} if summary.tpd is not None else {}),
        })

        # --- true MoE: [eval_model, mix_model], k=0 → 2*n samples ---
        print(f'\n  --- true MoE ({eval_config}+{mix_config}, k=0, S={2 * n}) ---')
        for m in [eval_model, mix_model]:
            setSeed(eval_cfg.seed)
            resetRng(m, eval_cfg.seed)
            m.eval()
        proposal = sampleMultiMoe(
            [eval_model, mix_model], eval_cfg, dl, device, k=0, seed=eval_cfg.seed
        )
        summary = getSummary(proposal, full_batch, calibrator=calibrator, likelihood_family=lf)
        rows.append({
            'config': eval_config,
            'condition': f'true MoE ({mix_config})',
            'total_samples': 2 * n,
            **{mn: ex(summary) for mn, ex, _ in METRICS},
            **({'t/ds': summary.tpd} if summary.tpd is not None else {}),
        })

    return rows


def evaluateMulti(
    configs: list[str],
    ks: list[int],
    eval_config: str | None,
    use_valid: bool,
    importance: bool,
    seeds: list[int] | None = None,
) -> list[dict]:
    """True MoE: load one checkpoint per config, mix proposals on shared eval data.

    Baselines: each individual model run separately.
    Mixture: all models combined with multiCheckpointEstimate.
    """
    if eval_config is None:
        eval_config = configs[0]

    partition = 'valid' if use_valid else 'test'

    print(f'\n{"=" * 60}')
    print(f'True MoE: {len(configs)} checkpoint(s) → eval on {eval_config}')
    print(f'Configs : {configs}')
    print(f'Partition: {partition}, IS={importance}')
    print(f'{"=" * 60}')

    # load eval infrastructure from the designated eval config
    eval_cfg = loadEvalConfig(eval_config, plot=False, importance=importance)
    setSeed(eval_cfg.seed)
    device = setDevice(eval_cfg.device)

    eval_model, eval_data_cfg, eval_run = initModel(eval_cfg, device)
    cal = calibrate(eval_model, eval_cfg, eval_data_cfg, eval_run, device)

    dl = getDataloader(eval_data_cfg, partition, batch_size=1)
    full_batch = dl.fullBatch()
    if eval_cfg.rescale:
        full_batch = rescaleData(full_batch)

    # load all checkpoints; validate compatibility
    models: list[Approximator] = []
    labels: list[str] = []
    seed_overrides = seeds if seeds is not None else [None] * len(configs)
    for name, seed_override in zip(configs, seed_overrides):
        cfg = loadEvalConfig(name, plot=False, importance=importance)
        if seed_override is not None:
            cfg.seed = seed_override
        lf = getattr(cfg, 'likelihood_family', 0)
        eval_lf = getattr(eval_cfg, 'likelihood_family', 0)
        if lf != eval_lf:
            raise ValueError(
                f'Config {name} has likelihood_family={lf} but eval config has {eval_lf}'
            )
        model, _, _ = initModel(cfg, device)
        models.append(model)
        label = f'{name}@s{cfg.seed}' if seed_override is not None else name
        labels.append(label)

    calibrator = cal if eval_cfg.conformal else None
    lf = getattr(eval_cfg, 'likelihood_family', 0)
    rows = []

    for k in ks:
        # individual baselines
        for model, label in zip(models, labels):
            print(f'\n  --- {label} (single, k={k}) ---')
            setSeed(eval_cfg.seed)
            resetRng(model, eval_cfg.seed)
            model.eval()
            proposal = sampleMoe(model, eval_cfg, dl, device, k, eval_cfg.seed)
            summary = getSummary(proposal, full_batch, calibrator=calibrator, likelihood_family=lf)
            row = {
                'config': eval_config,
                'condition': f'{label} (k={k})',
                'k': k,
                'total_samples': (1 + k) * eval_cfg.n_samples,
            }
            for metric_name, extractor, _ in METRICS:
                row[metric_name] = extractor(summary)
            if summary.tpd is not None:
                row['t/ds'] = summary.tpd
            rows.append(row)

        # mixture of all models
        total = len(models) * (1 + k) * eval_cfg.n_samples
        print(f'\n  --- mixture ({len(models)} experts, k={k}, {total} total samples) ---')
        for model in models:
            setSeed(eval_cfg.seed)
            resetRng(model, eval_cfg.seed)
            model.eval()
        proposal = sampleMultiMoe(models, eval_cfg, dl, device, k, eval_cfg.seed)
        summary = getSummary(proposal, full_batch, calibrator=calibrator, likelihood_family=lf)
        row = {
            'config': eval_config,
            'condition': f'mixture ({len(models)} experts, k={k})',
            'k': k,
            'total_samples': total,
        }
        for metric_name, extractor, _ in METRICS:
            row[metric_name] = extractor(summary)
        if summary.tpd is not None:
            row['t/ds'] = summary.tpd
        rows.append(row)

    return rows


def evaluate(
    configs: list[str],
    ks: list[int],
    use_valid: bool,
    importance: bool,
) -> list[dict]:
    """Run all configs × k values and collect metric rows."""
    rows = []
    partition = 'valid' if use_valid else 'test'

    for config_name in configs:
        print(f'\n{"=" * 60}')
        print(f'Config: {config_name} (partition={partition}, IS={importance})')
        print(f'{"=" * 60}')

        cfg = loadEvalConfig(config_name, plot=False)
        cfg.importance = importance
        setSeed(cfg.seed)
        device = setDevice(cfg.device)

        model, data_cfg, run = initModel(cfg, device)
        cal = calibrate(model, cfg, data_cfg, run, device)

        # B=1 dataloader (moe requires single datasets)
        dl = getDataloader(data_cfg, partition, batch_size=1)
        full_batch = dl.fullBatch()
        if cfg.rescale:
            full_batch = rescaleData(full_batch)

        # build run list: for each k, run MoE; for each k>0, also run a
        # control with k=0 but matched total sample count
        runs = []
        control_counts = set()
        for k in ks:
            total = (1 + k) * cfg.n_samples
            runs.append((k, cfg.n_samples, f'k={k}', total))
            if k > 0 and total not in control_counts:
                control_counts.add(total)
                runs.append((0, total, f'k=0 (S={total})', total))

        # sort: baseline first, then by total samples, controls before MoE
        runs.sort(key=lambda r: (r[3], r[0]))

        calibrator = cal if cfg.conformal else None
        lf = getattr(cfg, 'likelihood_family', 0)

        for k, n_samp, label, total in runs:
            print(f'\n  --- {label} ({1 + k} views, {total} total samples) ---')

            setSeed(cfg.seed)
            resetRng(model, cfg.seed)
            model.eval()

            # temporarily override n_samples for control conditions
            orig_n_samples = cfg.n_samples
            cfg.n_samples = n_samp
            proposal = sampleMoe(model, cfg, dl, device, k, cfg.seed)
            cfg.n_samples = orig_n_samples

            summary = getSummary(
                proposal, full_batch, calibrator=calibrator, likelihood_family=lf
            )

            row = {
                'config': config_name,
                'condition': label,
                'k': k,
                'total_samples': total,
            }
            for metric_name, extractor, _ in METRICS:
                row[metric_name] = extractor(summary)
            if summary.tpd is not None:
                row['t/ds'] = summary.tpd
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Table formatting and output
# ---------------------------------------------------------------------------


def bestIndices(rows: list[dict]) -> dict[str, set[int]]:
    """Find the row index of the best value per metric, grouped by config."""
    metric_names = [m[0] for m in METRICS]
    direction = {m[0]: m[2] for m in METRICS}
    best: dict[str, set[int]] = {m: set() for m in metric_names}

    configs = []
    seen = set()
    for r in rows:
        if r['config'] not in seen:
            configs.append(r['config'])
            seen.add(r['config'])

    for cfg in configs:
        cfg_rows = [(i, r) for i, r in enumerate(rows) if r['config'] == cfg]
        for metric in metric_names:
            values = [(i, r[metric]) for i, r in cfg_rows if r[metric] is not None]
            if not values:
                continue
            d = direction[metric]
            if d == 'abs':
                best_idx = min(values, key=lambda x: abs(x[1]))[0]
            elif d:
                best_idx = max(values, key=lambda x: x[1])[0]
            else:
                best_idx = min(values, key=lambda x: x[1])[0]
            best[metric].add(best_idx)

    return best


def deltaTable(rows: list[dict]) -> str:
    """Format a table showing changes from baseline (k=0) per config."""
    metric_names = [m[0] for m in METRICS]
    table_rows = []

    configs = []
    seen = set()
    for r in rows:
        if r['config'] not in seen:
            configs.append(r['config'])
            seen.add(r['config'])

    for cfg in configs:
        cfg_rows = [r for r in rows if r['config'] == cfg]
        baseline = cfg_rows[0]
        for r in cfg_rows[1:]:
            table_row = [cfg, r['condition'], r['total_samples']]
            for metric in metric_names:
                b_val = baseline[metric]
                val = r[metric]
                if b_val is None or val is None:
                    table_row.append('—')
                else:
                    delta = val - b_val
                    table_row.append(f'{delta:+.4f}')
            table_rows.append(table_row)

    headers = ['Config', 'Condition', 'Samples'] + [f'Δ{m}' for m in metric_names]
    return tabulate(table_rows, headers=headers, tablefmt='pipe', stralign='right')


def formatTable(rows: list[dict], fmt: str = 'pipe') -> str:
    """Format as a table with best-per-column markers."""
    metric_names = [m[0] for m in METRICS]
    best = bestIndices(rows)

    table_rows = []
    for i, r in enumerate(rows):
        table_row = [r['config'], r['condition'], r['total_samples']]
        for metric in metric_names:
            val = r[metric]
            if val is None:
                cell = '—'
            else:
                cell = f'{val:.4f}'
                if i in best.get(metric, set()):
                    if fmt == 'latex':
                        cell = f'\\textbf{{{cell}}}'
                    else:
                        cell = f'**{cell}**'
            table_row.append(cell)
        if 't/ds' in r:
            table_row.append(f'{r["t/ds"]:.3f}')
        table_rows.append(table_row)

    headers = ['Config', 'Condition', 'Samples'] + metric_names
    if any('t/ds' in r for r in rows):
        headers.append('t/ds [s]')
    tablefmt = 'latex_booktabs' if fmt == 'latex' else 'pipe'
    return tabulate(table_rows, headers=headers, tablefmt=tablefmt, stralign='right')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)

    ks = sorted(args.ks)

    if args.compare:
        if not args.mix_configs or len(args.mix_configs) != len(args.configs):
            raise ValueError('--compare requires --mix-configs with the same number of entries as --configs')
        print(f'MoE comparison: {len(args.configs)} family(ies), pseudo k=1 vs true 2-expert')
        print(f'Eval configs : {args.configs}')
        print(f'Mix  configs : {args.mix_configs}')
        print(f'Partition    : {"valid" if args.valid else "test"}')
        print(f'IS           : {args.importance}')
        rows = evaluateComparison(args.configs, args.mix_configs, args.valid, args.importance)
        outfile = 'compare_moe'
    elif args.multi:
        multi_configs = args.configs
        multi_seeds = args.seeds
        if multi_seeds is not None:
            if len(multi_configs) == 1:
                multi_configs = multi_configs * len(multi_seeds)
            elif len(multi_configs) != len(multi_seeds):
                raise ValueError(
                    f'--seeds length ({len(multi_seeds)}) must match --configs length '
                    f'({len(multi_configs)}) or --configs must have exactly one entry'
                )
        if len(multi_configs) < 2:
            raise ValueError('--multi requires at least 2 checkpoints (use --seeds to expand a single config)')
        print(f'True MoE experiment: {len(multi_configs)} checkpoint(s), {len(ks)} k value(s)')
        print(f'Checkpoints : {multi_configs}')
        if multi_seeds is not None:
            print(f'Seeds       : {multi_seeds}')
        print(f'Eval config : {args.eval_config or multi_configs[0]}')
        print(f'k values    : {ks}')
        print(f'Partition   : {"valid" if args.valid else "test"}')
        print(f'IS          : {args.importance}')
        rows = evaluateMulti(
            multi_configs, ks, args.eval_config, args.valid, args.importance, seeds=multi_seeds
        )
        outfile = 'multi_moe'
    else:
        print(f'Pseudo-MoE experiment: {len(args.configs)} config(s) × {len(ks)} k values')
        print(f'Configs: {args.configs}')
        print(f'k values: {ks}')
        print(f'Partition: {"valid" if args.valid else "test"}')
        print(f'IS: {args.importance}')
        rows = evaluate(args.configs, ks, args.valid, args.importance)
        outfile = 'moe'

    print(f'\n{formatTable(rows)}')
    print(f'\n{deltaTable(rows)}')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f'{outfile}.md'
    md_path.write_text(
        f'# MoE Results\n\n## Absolute\n\n{formatTable(rows)}\n\n'
        f'## Change from Baseline\n\n{deltaTable(rows)}\n'
    )
    tex_path = outdir / f'{outfile}.tex'
    tex_path.write_text(formatTable(rows, fmt='latex') + '\n')
    print(f'\nMarkdown → {md_path}')
    print(f'LaTeX    → {tex_path}')
    print('\nDone.')
