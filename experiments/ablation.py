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
    uv run python ablation.py --configs toy
    uv run python ablation.py --configs small-n-mixed large-n-mixed
"""

import argparse
import time
import yaml
from pathlib import Path

import torch
from tabulate import tabulate
from tqdm import tqdm

from metabeta.evaluation.evaluate import Evaluator
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.utils.evaluation import Proposal, concatProposalsBatch, dictMean
from metabeta.utils.io import datasetFilename
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.logger import setupLogging
from metabeta.posthoc.importance import runIS, runSIR
from metabeta.evaluation.summary import getSummary

DIR = Path(__file__).resolve().parent
EVAL_CFG_DIR = DIR / '..' / 'metabeta' / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results'

# default configs to evaluate (each must have a YAML in evaluation/configs/)
DEFAULT_CONFIGS = ['toy']

# (label, importance, sir, conformal)
CONDITIONS = [
    ('Baseline', False, False, False),
    ('+ IS', True, False, False),
    ('+ Conformal', False, False, True),
    ('+ IS + Conformal', True, False, True),
]

# (display name, extractor, higher_is_better)
METRICS = [
    ('R', lambda s: dictMean(s.corr), True),
    ('NRMSE', lambda s: dictMean(s.nrmse), False),
    ('ECE', lambda s: dictMean(s.ece), False),
    ('ppNLL', lambda s: s.mnll, False),
    ('R²', lambda s: s.mfit, True),
    ('t/ds', lambda s: s.tpd, False),
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


@torch.inference_mode()
def sampleMinibatched(
    evaluator: Evaluator,
    dl_test,
    label: str,
) -> Proposal:
    """Sample from proposal distribution over minibatches (like Trainer.sample)."""
    proposals = []
    n_datasets = 0
    t0 = time.perf_counter()
    for batch in tqdm(dl_test, desc=f'  {label}'):
        batch = toDevice(batch, evaluator.device)
        cfg = evaluator.cfg
        if cfg.importance and not cfg.sir:
            proposal = runIS(evaluator.model, batch, cfg)
        elif cfg.sir:
            proposal = runSIR(evaluator.model, batch, cfg)
        else:
            proposal = evaluator.model.estimate(batch, n_samples=cfg.n_samples)
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

        # init evaluator with conformal=True so validation data is loaded for calibration
        cfg = loadEvalConfig(config_name, conformal=True, plot=False)
        evaluator = Evaluator(cfg)

        # calibrate once
        calibrator = evaluator.calibrate()

        # create a batched test dataloader (prevents OOM for large models)
        data_fname = datasetFilename(evaluator.data_cfg, 'test')
        data_path = Path(evaluator.dir, '..', 'outputs', 'data', data_fname)
        assert data_path.exists(), f'test data not found: {data_path}'
        dl_test = Dataloader(data_path, batch_size=batch_size, sortish=True)

        # get full test batch for evaluation (collated on CPU)
        full_batch = dl_test.fullBatch()
        if evaluator.cfg.rescale:
            full_batch = rescaleData(full_batch)

        for label, importance, sir, conformal in CONDITIONS:
            print(f'\n  --- {label} ---')

            # toggle post-hoc flags
            evaluator.cfg.importance = importance
            evaluator.cfg.sir = sir

            # sample in minibatches
            proposal = sampleMinibatched(evaluator, dl_test, label)

            # summarize on full batch
            cal = calibrator if conformal else None
            lf = getattr(evaluator.cfg, 'likelihood_family', 0)
            summary = getSummary(proposal, full_batch, calibrator=cal, likelihood_family=lf)

            # extract metrics
            row = {'config': config_name, 'condition': label}
            for metric_name, extractor, _ in METRICS:
                row[metric_name] = extractor(summary)
            rows.append(row)

    return rows


def bestIndices(rows: list[dict]) -> dict[str, set[int]]:
    """Find the row index of the best value per metric, grouped by config."""
    metric_names = [m[0] for m in METRICS]
    higher_is_better = {m[0]: m[2] for m in METRICS}
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
            if higher_is_better[metric]:
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
