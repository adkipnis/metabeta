"""
Ablation study: evaluate trained models under different post-hoc settings.

For each evaluation config (= trained model):
    1. Load model from checkpoint
    2. Load test set (and validation set for calibration)
    3. Calibrate conformal intervals on validation set
    4. Evaluate on test set under 4 conditions:
        - Baseline
        - Baseline + IS
        - Baseline + Conformal
        - Baseline + IS + Conformal
    5. Collect metrics: R, NRMSE, ECE, ppNLL, R², time/dataset
    6. Mark best entry per column (per config)
    7. Save table in markdown and LaTeX format

Usage (from experiments/):
    uv run python ablation.py
    uv run python ablation.py --configs toy-n small-n-sampled
    uv run python ablation.py --batch_size 4
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
from metabeta.posthoc.importance import runIS, runSIR
from metabeta.utils.config import (
    ApproximatorConfig,
    assimilateConfig,
    loadDataConfig,
    modelFromYaml,
)
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.evaluation import Proposal, concatProposalsBatch, dictMean
from metabeta.utils.io import datasetFilename, runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.evaluation.summary import getSummary

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results'

# default configs to evaluate (each must have a YAML in evaluation/configs/)
DEFAULT_CONFIGS = ['toy-n', 'small-n-sampled']

# (label, importance, sir, conformal)
CONDITIONS = [
    ('Baseline', False, False, False),
    ('+ Conformal', False, False, True),
    ('+ IS', True, False, False),
    ('+ Conformal + IS', True, False, True),
]

# (display name, extractor, higher_is_better)
# higher_is_better: True = max, False = min, 'abs' = closest to 0
METRICS = [
    ('R', lambda s: dictMean(s.corr), True),
    ('NRMSE', lambda s: dictMean(s.nrmse), False),
    ('ECE', lambda s: dictMean(s.ece), 'abs'),
    ('ppNLL', lambda s: s.mnll, False),
    ('R²', lambda s: s.mfit, True),
]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ablation study across post-hoc methods.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names (YAML files in evaluation/configs/)')
    parser.add_argument('--batch_size', type=int, default=8, help='minibatch size for sampling (prevents OOM)')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR), help='output directory for tables')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
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
    # data config (needed for d/q dimensions)
    data_cfg = loadDataConfig(cfg.d_tag)
    assimilateConfig(cfg, data_cfg)

    # model config
    model_cfg_path = METABETA / 'models' / 'configs' / f'{cfg.m_tag}.yaml'
    model_cfg = modelFromYaml(
        model_cfg_path,
        d_ffx=cfg.max_d,
        d_rfx=cfg.max_q,
        likelihood_family=getattr(cfg, 'likelihood_family', 0),
    )
    model = Approximator(model_cfg).to(device)
    model.eval()

    # load checkpoint
    run = runName(vars(cfg))
    ckpt_dir = METABETA / 'outputs' / 'checkpoints' / run
    path = ckpt_dir / f'{cfg.prefix}.pt'
    assert path.exists(), f'checkpoint not found: {path}'
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload['model_state'])

    if cfg.compile and device.type != 'mps':
        model.compile()

    return model, data_cfg, run


def getDataloader(data_cfg: dict, partition: str, batch_size: int | None = None) -> Dataloader:
    """Create a dataloader for the given partition."""
    data_fname = datasetFilename(data_cfg, partition)
    data_path = METABETA / 'outputs' / 'data' / data_fname
    assert data_path.exists(), f'data not found: {data_path}'
    sortish = batch_size is not None
    return Dataloader(data_path, batch_size=batch_size, sortish=sortish)


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


def resetRng(model: Approximator, seed: int) -> None:
    """Reset base distribution RNGs for reproducible sampling."""
    model.posterior_g.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore
    model.posterior_l.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore


@torch.inference_mode()
def sampleMinibatched(
    model: Approximator,
    cfg: argparse.Namespace,
    dl: Dataloader,
    device: torch.device,
    label: str,
) -> Proposal:
    """Sample from proposal distribution over minibatches (like Trainer.sample)."""
    proposals = []
    n_datasets = 0
    t0 = time.perf_counter()
    for batch in tqdm(dl, desc=f'  {label}'):
        batch = toDevice(batch, device)
        if cfg.importance and not cfg.sir:
            proposal = runIS(model, batch, cfg)
        elif cfg.sir:
            proposal = runSIR(model, batch, cfg)
        else:
            proposal = model.estimate(batch, n_samples=cfg.n_samples)
            if cfg.rescale:
                proposal.rescale(batch['sd_y'])
        proposal.to('cpu')
        proposals.append(proposal)
        n_datasets += batch['X'].shape[0]
    t1 = time.perf_counter()

    merged = concatProposalsBatch(proposals)
    merged.tpd = (t1 - t0) / max(n_datasets, 1)
    return merged


def evaluate(configs: list[str], batch_size: int) -> list[dict]:
    """Run all configs × conditions and collect metric rows."""
    rows = []

    for config_name in configs:
        print(f'\n{"=" * 60}')
        print(f'Config: {config_name}')
        print(f'{"=" * 60}')

        cfg = loadEvalConfig(config_name, plot=False)
        setSeed(cfg.seed)
        device = setDevice(cfg.device)

        # load model and data config
        model, data_cfg, run = initModel(cfg, device)

        # calibrate once on validation set
        cal = calibrate(model, cfg, data_cfg, run, device)

        # create batched test dataloader (prevents OOM for large models)
        dl_test = getDataloader(data_cfg, 'test', batch_size=batch_size)

        # get full test batch for evaluation metrics (collated on CPU)
        full_batch = dl_test.fullBatch()
        if cfg.rescale:
            full_batch = rescaleData(full_batch)

        for label, importance, sir, conformal in CONDITIONS:
            print(f'\n  --- {label} ---')

            # toggle post-hoc flags
            cfg.importance = importance
            cfg.sir = sir

            # reset RNGs so base distribution draws are identical across conditions
            setSeed(cfg.seed)
            resetRng(model, cfg.seed)
            model.eval()

            # sample in minibatches
            proposal = sampleMinibatched(model, cfg, dl_test, device, label)

            # summarize on full batch
            calibrator = cal if conformal else None
            lf = getattr(cfg, 'likelihood_family', 0)
            summary = getSummary(proposal, full_batch, calibrator=calibrator, likelihood_family=lf)

            # extract metrics
            row = {'config': config_name, 'condition': label}
            for metric_name, extractor, _ in METRICS:
                row[metric_name] = extractor(summary)
            rows.append(row)

    return rows


def bestIndices(rows: list[dict]) -> dict[str, set[int]]:
    """Find the row index of the best value per metric, grouped by config.

    higher_is_better: True = max, False = min, 'abs' = closest to 0.
    """
    metric_names = [m[0] for m in METRICS]
    direction = {m[0]: m[2] for m in METRICS}
    best: dict[str, set[int]] = {m: set() for m in metric_names}

    # group rows by config
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


def formatTable(rows: list[dict], fmt: str = 'pipe') -> str:
    """Format as a table with best-per-column markers.

    fmt: 'pipe' for markdown, 'latex' for LaTeX.
    """
    metric_names = [m[0] for m in METRICS]
    best = bestIndices(rows)

    table_rows = []
    for i, r in enumerate(rows):
        table_row = [r['config'], r['condition']]
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
        table_rows.append(table_row)

    headers = ['Config', 'Condition'] + metric_names
    tablefmt = 'latex_booktabs' if fmt == 'latex' else 'pipe'
    return tabulate(table_rows, headers=headers, tablefmt=tablefmt, stralign='right')


def save(rows: list[dict], outdir: Path) -> None:
    """Save tables in markdown and LaTeX formats."""
    outdir.mkdir(parents=True, exist_ok=True)

    md_table = formatTable(rows, fmt='pipe')
    md_path = outdir / 'ablation.md'
    md_path.write_text(f'# Ablation Results\n\n{md_table}\n')
    print(f'\nMarkdown saved to {md_path}')

    tex_table = formatTable(rows, fmt='latex')
    tex_path = outdir / 'ablation.tex'
    tex_path.write_text(tex_table + '\n')
    print(f'LaTeX saved to {tex_path}')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)

    print(f'Ablation study: {len(args.configs)} config(s) × {len(CONDITIONS)} conditions')
    print(f'Configs: {args.configs}')

    rows = evaluate(args.configs, args.batch_size)

    # print to console
    print(f'\n{formatTable(rows)}')

    # save
    save(rows, Path(args.outdir))
    print('\nDone.')
