"""
Misspecification study: evaluate trained models under prior misspecification.

For each evaluation config:
    1. Load model from checkpoint
    2. Load test set (or validation set under --valid flag)
    3. For each condition (baseline + misspecified):
        - Perturb prior context in the batch (scale, location, family)
        - Sample from the model
        - Evaluate metrics against the same ground truth
    4. Summarize absolute metrics and changes from baseline
    5. Save table in markdown and LaTeX format

Perturbation types:
    - Wrong variance: scale tau hyperparameters (multiple factors supported)
    - Wrong mean: shift nu_ffx by k × tau_ffx (relative to prior width)
    - Wrong family: rotate prior family indices (+1 mod n_families)

Usage (from experiments/):
    uv run python missspecification.py
    uv run python missspecification.py --configs toy-n
    uv run python missspecification.py --scale_factors 0.33 3 10
    uv run python missspecification.py --mean_shifts 1 2 5
    uv run python missspecification.py --valid
"""
# subset datatsets

import argparse
import yaml
from dataclasses import dataclass
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
from metabeta.utils.families import FFX_FAMILIES, SIGMA_FAMILIES
from metabeta.posthoc.importance import runIS
from metabeta.utils.io import datasetFilename, runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed
from metabeta.evaluation.summary import getSummary

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results'

DEFAULT_CONFIGS = ['toy-n']
DEFAULT_SCALE_FACTORS = [0.33, 3.0]
DEFAULT_MEAN_SHIFTS = [1.0, 2.0]

# (display name, extractor, higher_is_better)
# higher_is_better: True = max, False = min, 'abs' = closest to 0
METRICS = [
    ('R', lambda s: dictMean(s.corr), True),
    ('NRMSE', lambda s: dictMean(s.nrmse), False),
    ('ECE', lambda s: dictMean(s.ece), 'abs'),
    ('ppNLL', lambda s: s.mnll, False),
]


@dataclass
class Condition:
    label: str
    scale_factor: float = 1.0   # tau multiplier (1.0 = no change)
    mean_shift: float = 0.0     # nu_ffx offset in units of tau_ffx (0.0 = no change)
    wrong_family: bool = False


def buildConditions(
    scale_factors: list[float], mean_shifts: list[float]
) -> list[Condition]:
    """Build all combinations of (scale, shift, family) perturbations."""
    scales = [1.0] + sorted(scale_factors)
    shifts = [0.0] + sorted(mean_shifts)
    families = [False, True]

    conditions = []
    for s in scales:
        for k in shifts:
            for fam in families:
                parts = []
                if s != 1.0:
                    parts.append(f'τ×{s:g}')
                if k != 0.0:
                    parts.append(f'μ+{k:g}σ')
                if fam:
                    parts.append('fam')
                label = ' + '.join(parts) if parts else 'Baseline'
                conditions.append(Condition(label, s, k, fam))

    return conditions


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Misspecification study across prior perturbations.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names (YAML files in evaluation/configs/)')
    parser.add_argument('--scale_factors', nargs='+', type=float, default=DEFAULT_SCALE_FACTORS, help='tau multipliers for wrong-variance conditions')
    parser.add_argument('--mean_shifts', nargs='+', type=float, default=DEFAULT_MEAN_SHIFTS, help='nu_ffx offsets in units of tau_ffx for wrong-mean conditions')
    parser.add_argument('--importance', action='store_true', help='use importance sampling post-hoc')
    parser.add_argument('--batch_size', type=int, default=8, help='minibatch size for sampling (prevents OOM)')
    parser.add_argument('--valid', action='store_true', help='use validation set instead of test set')
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


# ---------------------------------------------------------------------------
# Batch perturbation
# ---------------------------------------------------------------------------


def perturbBatch(
    batch: dict[str, torch.Tensor],
    cond: Condition,
) -> dict[str, torch.Tensor]:
    """Clone batch and perturb prior context fields.

    scale_factor: multiply tau_ffx, tau_rfx, and tau_eps by this factor.
    mean_shift: add mean_shift × tau_ffx to nu_ffx (relative to prior width).
    wrong_family: rotate family indices (+1 mod n_families).
    """
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    if cond.scale_factor != 1.0:
        out['tau_ffx'] = out['tau_ffx'] * cond.scale_factor
        out['tau_rfx'] = out['tau_rfx'] * cond.scale_factor
        if 'tau_eps' in out:
            out['tau_eps'] = out['tau_eps'] * cond.scale_factor

    if cond.mean_shift != 0.0:
        out['nu_ffx'] = out['nu_ffx'] + cond.mean_shift * batch['tau_ffx']

    if cond.wrong_family:
        n_ffx = len(FFX_FAMILIES)
        n_sigma = len(SIGMA_FAMILIES)
        out['family_ffx'] = (out['family_ffx'] + 1) % n_ffx
        out['family_sigma_rfx'] = (out['family_sigma_rfx'] + 1) % n_sigma
        if 'family_sigma_eps' in out:
            out['family_sigma_eps'] = (out['family_sigma_eps'] + 1) % n_sigma

    return out


# ---------------------------------------------------------------------------
# Sampling and evaluation
# ---------------------------------------------------------------------------


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
def sampleMinibatched(
    model: Approximator,
    cfg: argparse.Namespace,
    dl: Dataloader,
    device: torch.device,
    cond: Condition,
) -> Proposal:
    """Sample from proposal distribution over minibatches with optional perturbation."""
    proposals = []
    for batch in tqdm(dl, desc=f'  {cond.label}'):
        batch = toDevice(batch, device)
        batch = perturbBatch(batch, cond)
        if cfg.importance:
            proposal = runIS(model, batch, cfg)
        else:
            proposal = model.estimate(batch, n_samples=cfg.n_samples)
            if cfg.rescale:
                proposal.rescale(batch['sd_y'])
        proposal.to('cpu')
        proposals.append(proposal)

    return concatProposalsBatch(proposals)


def evaluate(
    configs: list[str],
    conditions: list[Condition],
    batch_size: int,
    use_valid: bool,
    importance: bool,
) -> list[dict]:
    """Run all configs × conditions and collect metric rows."""
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

        dl = getDataloader(data_cfg, partition, batch_size=batch_size)
        full_batch = dl.fullBatch()
        if cfg.rescale:
            full_batch = rescaleData(full_batch)

        for cond in conditions:
            print(f'\n  --- {cond.label} ---')

            setSeed(cfg.seed)
            resetRng(model, cfg.seed)
            model.eval()

            proposal = sampleMinibatched(model, cfg, dl, device, cond)

            calibrator = cal if cfg.conformal else None
            lf = getattr(cfg, 'likelihood_family', 0)
            summary = getSummary(
                proposal, full_batch, calibrator=calibrator, likelihood_family=lf
            )

            row = {'config': config_name, 'condition': cond.label}
            for metric_name, extractor, _ in METRICS:
                row[metric_name] = extractor(summary)
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Table formatting and output
# ---------------------------------------------------------------------------


def bestIndices(rows: list[dict]) -> dict[str, set[int]]:
    """Find the row index of the best value per metric, grouped by config.

    higher_is_better: True = max, False = min, 'abs' = closest to 0.
    """
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
    """Format a table showing changes from baseline per config."""
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
            table_row = [cfg, r['condition']]
            for metric in metric_names:
                b_val = baseline[metric]
                val = r[metric]
                if b_val is None or val is None:
                    table_row.append('—')
                else:
                    delta = val - b_val
                    table_row.append(f'{delta:+.4f}')
            table_rows.append(table_row)

    headers = ['Config', 'Condition'] + [f'Δ{m}' for m in metric_names]
    return tabulate(table_rows, headers=headers, tablefmt='pipe', stralign='right')


def formatTable(rows: list[dict], fmt: str = 'pipe') -> str:
    """Format as a table with best-per-column markers."""
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

    md_abs = formatTable(rows, fmt='pipe')
    md_delta = deltaTable(rows)
    md_path = outdir / 'missspecification.md'
    md_path.write_text(
        f'# Misspecification Results\n\n## Absolute\n\n{md_abs}\n\n'
        f'## Change from Baseline\n\n{md_delta}\n'
    )
    print(f'\nMarkdown saved to {md_path}')

    tex_table = formatTable(rows, fmt='latex')
    tex_path = outdir / 'missspecification.tex'
    tex_path.write_text(tex_table + '\n')
    print(f'LaTeX saved to {tex_path}')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)

    conditions = buildConditions(args.scale_factors, args.mean_shifts)

    print(f'Misspecification study: {len(args.configs)} config(s) × {len(conditions)} conditions')
    print(f'Configs: {args.configs}')
    print(f'Scale factors: {args.scale_factors}')
    print(f'Mean shifts: {args.mean_shifts}')
    print(f'Partition: {"valid" if args.valid else "test"}')
    print(f'IS: {args.importance}')
    print(f'Conditions: {[c.label for c in conditions]}')

    rows = evaluate(args.configs, conditions, args.batch_size, args.valid, args.importance)

    print(f'\n{formatTable(rows)}')
    print(f'\n{deltaTable(rows)}')

    save(rows, Path(args.outdir))
    print('\nDone.')
