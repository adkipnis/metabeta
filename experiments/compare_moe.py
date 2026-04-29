"""
Mixture of Experts experiment: pseudo (permutation) and true (multi-checkpoint) modes.

Pseudo-MoE (default): single checkpoint, k random feature permutations per dataset.
True MoE (--multi)  : one checkpoint per config, proposals mixed across all models.

Usage (from experiments/):
    uv run python compare_moe.py --configs normal_dsmall-n-sampled_mlarge_s0 --ks 0 3 7
    uv run python compare_moe.py --multi --configs normal_dsmall-n-mixed_mlarge --seeds 0 1 2 3
    uv run python compare_moe.py --multi --configs normal_dsmall-n-mixed_mlarge_s0 normal_dsmall-n-mixed_mlarge_s1
    uv run python compare_moe.py --compare --configs normal_dsmall-n-mixed_mlarge_s0 --mix-configs normal_dsmall-n-mixed_mlarge_s1
    uv run python compare_moe.py --valid
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
from metabeta.utils.config import (
    assimilateConfig,
    loadDataConfig,
    modelFromYaml,
)
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.evaluation import Proposal, concatProposalsBatch, dictMean
from metabeta.utils.io import datasetFilename, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.moe import moeEstimate, multiCheckpointEstimate
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.evaluation.summary import getSummary

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
OUT_DIR = DIR / 'results'

DEFAULT_CONFIGS = ['normal_dtiny-n-toy_mtiny_s42']
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
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='checkpoint run names (directories under outputs/checkpoints/)')
    parser.add_argument('--ks', nargs='+', type=int, default=DEFAULT_KS, help='number of extra permuted views (0 = baseline)')
    parser.add_argument('--valid', action='store_true', help='use validation set instead of test set')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR), help='output directory for tables')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    # multi-checkpoint MoE mode
    parser.add_argument('--multi', action='store_true', help='true MoE: treat --configs as a list of checkpoints to mix')
    parser.add_argument('--eval-config', type=str, default=None, help='full run name used for data/settings in --multi mode (default: first resolved run name)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='seeds for --multi mode; a single --configs entry is paired with each seed as {config}_s{seed}')
    # head-to-head comparison mode
    parser.add_argument('--compare', action='store_true', help='compare pseudo-MoE (k=1) vs true MoE (2 experts) at matched sample counts')
    parser.add_argument('--mix-configs', nargs='+', default=None, help='second checkpoint run name per eval config for --compare mode (must match --configs length)')
    return parser.parse_args()
# fmt: on


def resolveRunName(base: str, seed: int | None) -> str:
    """Append _s{seed} suffix to a base run name when seed is given."""
    return f'{base}_s{seed}' if seed is not None else base


def loadRunConfig(run_name: str) -> argparse.Namespace:
    """Load config from the checkpoint directory."""
    path = METABETA / 'outputs' / 'checkpoints' / run_name / 'config.yaml'
    assert path.exists(), f'checkpoint config not found: {path}'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = run_name
    cfg.setdefault('prefix', 'best')
    cfg.setdefault('device', '')
    return argparse.Namespace(**cfg)


def initModel(cfg: argparse.Namespace, device: torch.device) -> tuple[Approximator, dict, str]:
    """Load model architecture from checkpoint config and restore weights."""
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

    ckpt_dir = METABETA / 'outputs' / 'checkpoints' / cfg.run_name
    path = ckpt_dir / f'{cfg.prefix}.pt'
    assert path.exists(), f'checkpoint not found: {path}'
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload['model_state'])

    if cfg.compile and device.type != 'mps':
        model.compile()

    # Prefer data_id_valid for test/valid evaluation data (may differ from training data)
    eval_data_id = getattr(cfg, 'data_id_valid', None) or cfg.data_id
    eval_data_cfg = loadDataConfig(eval_data_id)

    return model, eval_data_cfg, cfg.run_name


def getDataloader(data_cfg: dict, partition: str, batch_size: int | None = None) -> Dataloader:
    """Create a dataloader for the given partition."""
    data_fname = datasetFilename(partition)
    data_path = METABETA / 'outputs' / 'data' / data_cfg['data_id'] / data_fname
    assert data_path.exists(), f'data not found: {data_path}'
    sortish = batch_size is not None
    return Dataloader(data_path, batch_size=batch_size, sortish=sortish)



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
    t0 = time.perf_counter()

    for batch in tqdm(dl, desc=f'  k={k}'):
        batch = toDevice(batch, device)
        B = batch['X'].shape[0]

        for i in range(B):
            single = {k_: v[i : i + 1] if torch.is_tensor(v) else v for k_, v in batch.items()}
            rng = np.random.default_rng(seed + n_datasets)
            proposal = moeEstimate(model, single, cfg.n_samples, k, rng=rng)
            if cfg.rescale:
                proposal.rescale(single['sd_y'])
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
    t0 = time.perf_counter()

    for batch in tqdm(dl, desc=f'  multi k={k}'):
        batch = toDevice(batch, device)
        B = batch['X'].shape[0]

        for i in range(B):
            single = {k_: v[i : i + 1] if torch.is_tensor(v) else v for k_, v in batch.items()}
            rng = np.random.default_rng(seed + n_datasets)
            proposal = multiCheckpointEstimate(models, single, cfg.n_samples, k=k, rng=rng)
            if cfg.rescale:
                proposal.rescale(single['sd_y'])
            proposal.to('cpu')
            proposals.append(proposal)
            n_datasets += 1

    t1 = time.perf_counter()
    merged = concatProposalsBatch(proposals)
    merged.tpd = (t1 - t0) / max(n_datasets, 1)
    return merged


def evaluateComparison(
    eval_runs: list[str],
    mix_runs: list[str],
    use_valid: bool,
) -> list[dict]:
    """Head-to-head: pseudo-MoE (k=1) vs true MoE (2 experts, k=0) at equal sample counts.

    For each (eval_run, mix_run) pair, runs three conditions — all on the
    eval run's test data:
      baseline   : eval model, k=0, n_samples       → n_samples total
      pseudo-MoE : eval model, k=1, n_samples       → 2*n_samples total
      true MoE   : [eval_model, mix_model], k=0     → 2*n_samples total
    """
    if len(eval_runs) != len(mix_runs):
        raise ValueError('--configs and --mix-configs must have the same length')

    partition = 'valid' if use_valid else 'test'
    rows = []

    for eval_run, mix_run in zip(eval_runs, mix_runs):
        print(f"\n{'=' * 60}")
        print(f'Comparison: {eval_run} vs +{mix_run}  (partition={partition})')
        print(f"{'=' * 60}")

        eval_cfg = loadRunConfig(eval_run)
        setSeed(eval_cfg.seed)
        device = setDevice(eval_cfg.device)

        eval_model, eval_data_cfg, _ = initModel(eval_cfg, device)
        lf = getattr(eval_cfg, 'likelihood_family', 0)

        dl = getDataloader(eval_data_cfg, partition, batch_size=1)
        full_batch = dl.fullBatch()
        if eval_cfg.rescale:
            full_batch = rescaleData(full_batch)

        mix_cfg = loadRunConfig(mix_run)
        if getattr(mix_cfg, 'likelihood_family', 0) != lf:
            raise ValueError(f'{mix_run} has different likelihood_family than {eval_run}')
        mix_model, _, _ = initModel(mix_cfg, device)

        n = eval_cfg.n_samples

        print(f'\n  --- baseline ({eval_run}, k=0, S={n}) ---')
        setSeed(eval_cfg.seed)
        eval_model.eval()
        proposal = sampleMoe(eval_model, eval_cfg, dl, device, k=0, seed=eval_cfg.seed)
        summary = getSummary(proposal, full_batch, likelihood_family=lf)
        rows.append(
            {
                'config': eval_run,
                'condition': 'baseline',
                'total_samples': n,
                **{mn: ex(summary) for mn, ex, _ in METRICS},
                **({'t/ds': summary.tpd} if summary.tpd is not None else {}),
            }
        )

        print(f'\n  --- pseudo-MoE ({eval_run}, k=1, S={2 * n}) ---')
        setSeed(eval_cfg.seed)
        eval_model.eval()
        proposal = sampleMoe(eval_model, eval_cfg, dl, device, k=1, seed=eval_cfg.seed)
        summary = getSummary(proposal, full_batch, likelihood_family=lf)
        rows.append(
            {
                'config': eval_run,
                'condition': 'pseudo-MoE (k=1)',
                'total_samples': 2 * n,
                **{mn: ex(summary) for mn, ex, _ in METRICS},
                **({'t/ds': summary.tpd} if summary.tpd is not None else {}),
            }
        )

        print(f'\n  --- true MoE ({eval_run}+{mix_run}, k=0, S={2 * n}) ---')
        for m in [eval_model, mix_model]:
            setSeed(eval_cfg.seed)
            m.eval()
        proposal = sampleMultiMoe(
            [eval_model, mix_model], eval_cfg, dl, device, k=0, seed=eval_cfg.seed
        )
        summary = getSummary(proposal, full_batch, likelihood_family=lf)
        rows.append(
            {
                'config': eval_run,
                'condition': f'true MoE (+{mix_run})',
                'total_samples': 2 * n,
                **{mn: ex(summary) for mn, ex, _ in METRICS},
                **({'t/ds': summary.tpd} if summary.tpd is not None else {}),
            }
        )

    return rows


def evaluateMulti(
    configs: list[str],
    ks: list[int],
    eval_run: str | None,
    use_valid: bool,
    seeds: list[int] | None = None,
) -> list[dict]:
    """True MoE: load one checkpoint per run, mix proposals on shared eval data.

    Baselines: each individual model run separately.
    Mixture: all models combined with multiCheckpointEstimate.
    """
    seed_list = seeds if seeds is not None else [None] * len(configs)
    bases = configs * len(seed_list) if len(configs) == 1 and len(seed_list) > 1 else configs
    run_names = [resolveRunName(base, seed) for base, seed in zip(bases, seed_list)]

    if eval_run is None:
        eval_run = run_names[0]

    partition = 'valid' if use_valid else 'test'

    print(f"\n{'=' * 60}")
    print(f'True MoE: {len(run_names)} checkpoint(s) → eval on {eval_run}')
    print(f'Run names: {run_names}')
    print(f'Partition: {partition}')
    print(f"{'=' * 60}")

    eval_cfg = loadRunConfig(eval_run)
    setSeed(eval_cfg.seed)
    device = setDevice(eval_cfg.device)

    eval_model, eval_data_cfg, _ = initModel(eval_cfg, device)

    dl = getDataloader(eval_data_cfg, partition, batch_size=1)
    full_batch = dl.fullBatch()
    if eval_cfg.rescale:
        full_batch = rescaleData(full_batch)

    models: list[Approximator] = []
    labels: list[str] = []
    eval_lf = getattr(eval_cfg, 'likelihood_family', 0)
    for run_name in run_names:
        cfg = loadRunConfig(run_name)
        lf = getattr(cfg, 'likelihood_family', 0)
        if lf != eval_lf:
            raise ValueError(
                f'Run {run_name} has likelihood_family={lf} but eval run has {eval_lf}'
            )
        model, _, _ = initModel(cfg, device)
        models.append(model)
        labels.append(run_name)

    lf = eval_lf
    rows = []

    for k in ks:
        for model, label in zip(models, labels):
            print(f'\n  --- {label} (single, k={k}) ---')
            setSeed(eval_cfg.seed)
            model.eval()
            proposal = sampleMoe(model, eval_cfg, dl, device, k, eval_cfg.seed)
            summary = getSummary(proposal, full_batch, likelihood_family=lf)
            row = {
                'config': eval_run,
                'condition': f'{label} (k={k})',
                'k': k,
                'total_samples': (1 + k) * eval_cfg.n_samples,
            }
            for metric_name, extractor, _ in METRICS:
                row[metric_name] = extractor(summary)
            if summary.tpd is not None:
                row['t/ds'] = summary.tpd
            rows.append(row)

        total = len(models) * (1 + k) * eval_cfg.n_samples
        print(f'\n  --- mixture ({len(models)} experts, k={k}, {total} total samples) ---')
        for model in models:
            setSeed(eval_cfg.seed)
            model.eval()
        proposal = sampleMultiMoe(models, eval_cfg, dl, device, k, eval_cfg.seed)
        summary = getSummary(proposal, full_batch, likelihood_family=lf)
        row = {
            'config': eval_run,
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
) -> list[dict]:
    """Run all run names × k values and collect metric rows."""
    rows = []
    partition = 'valid' if use_valid else 'test'

    for run_name in configs:
        print(f"\n{'=' * 60}")
        print(f'Run: {run_name} (partition={partition})')
        print(f"{'=' * 60}")

        cfg = loadRunConfig(run_name)
        setSeed(cfg.seed)
        device = setDevice(cfg.device)

        model, data_cfg, _ = initModel(cfg, device)

        dl = getDataloader(data_cfg, partition, batch_size=1)
        full_batch = dl.fullBatch()
        if cfg.rescale:
            full_batch = rescaleData(full_batch)

        runs = []
        control_counts = set()
        for k in ks:
            total = (1 + k) * cfg.n_samples
            runs.append((k, cfg.n_samples, f'k={k}', total))
            if k > 0 and total not in control_counts:
                control_counts.add(total)
                runs.append((0, total, f'k=0 (S={total})', total))

        runs.sort(key=lambda r: (r[3], r[0]))

        lf = getattr(cfg, 'likelihood_family', 0)

        for k, n_samp, label, total in runs:
            print(f'\n  --- {label} ({1 + k} views, {total} total samples) ---')

            setSeed(cfg.seed)
            model.eval()

            orig_n_samples = cfg.n_samples
            cfg.n_samples = n_samp
            proposal = sampleMoe(model, cfg, dl, device, k, cfg.seed)
            cfg.n_samples = orig_n_samples

            summary = getSummary(proposal, full_batch, likelihood_family=lf)

            row = {
                'config': run_name,
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
            table_row.append(f"{r['t/ds']:.3f}")
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
            raise ValueError(
                '--compare requires --mix-configs with the same number of entries as --configs'
            )
        print(f'MoE comparison: {len(args.configs)} pair(s), pseudo k=1 vs true 2-expert')
        print(f'Eval runs : {args.configs}')
        print(f'Mix  runs : {args.mix_configs}')
        print(f"Partition : {'valid' if args.valid else 'test'}")
        rows = evaluateComparison(args.configs, args.mix_configs, args.valid)
        outfile = 'compare_moe'
    elif args.multi:
        multi_seeds = args.seeds
        if multi_seeds is not None and len(args.configs) > 1 and len(args.configs) != len(multi_seeds):
            raise ValueError(
                f'--seeds length ({len(multi_seeds)}) must match --configs length '
                f'({len(args.configs)}) or --configs must have exactly one entry'
            )
        n_ckpts = len(multi_seeds) if multi_seeds else len(args.configs)
        if n_ckpts < 2:
            raise ValueError(
                '--multi requires at least 2 checkpoints (use --seeds to expand a single config)'
            )
        print(f'True MoE experiment: {n_ckpts} checkpoint(s), {len(ks)} k value(s)')
        print(f'Base configs: {args.configs}')
        if multi_seeds:
            print(f'Seeds       : {multi_seeds}')
        print(f'k values    : {ks}')
        print(f"Partition   : {'valid' if args.valid else 'test'}")
        rows = evaluateMulti(
            args.configs,
            ks,
            args.eval_config,
            args.valid,
            seeds=multi_seeds,
        )
        outfile = 'multi_moe'
    else:
        print(f'Pseudo-MoE experiment: {len(args.configs)} run(s) × {len(ks)} k values')
        print(f'Run names: {args.configs}')
        print(f'k values : {ks}')
        print(f"Partition: {'valid' if args.valid else 'test'}")
        rows = evaluate(args.configs, ks, args.valid)
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
