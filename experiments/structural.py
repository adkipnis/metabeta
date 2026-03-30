"""
Structural misspecification: metabeta vs NUTS under model misspecification.

Generate datasets from the standard simulation pipeline and add a mean-centered
nonlinear term (a scaled squared predictor) to y, varying the scale of the
nonlinearity.  Both metabeta and NUTS fit the *incorrectly specified* linear
model.  The key question is not parameter recovery (ground truth is no longer
well-defined) but whether metabeta faithfully approximates the posterior of the
misspecified linear model — the same model NUTS fits.  Agreement between the two
methods demonstrates that metabeta's errors are due to model choice, not the
inference procedure.

Workflow:
    1. Load model checkpoint and validation data
    2. For each nonlinearity scale (0 = baseline, then increasing):
        a. Perturb y by adding scale × mean-centered X² to each dataset
        b. Run metabeta on the perturbed data
        c. Run NUTS on the perturbed data (per-dataset, using Bambi)
        d. Compare posteriors: correlation and RMSE of posterior means
    3. Save results table

Usage (from experiments/):
    uv run python structural.py
    uv run python structural.py --configs small-n-sampled
    uv run python structural.py --scales 0.0 0.25 0.5 1.0 2.0
    uv run python structural.py --max_datasets 8 --nuts_draws 500
"""

import argparse
import time
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm

import bambi as bmb
import arviz as az

from metabeta.models.approximator import Approximator
from metabeta.utils.config import (
    assimilateConfig,
    loadDataConfig,
    modelFromYaml,
)
from metabeta.utils.dataloader import Dataloader, collateGrouped, toDevice
from metabeta.utils.evaluation import Proposal
from metabeta.utils.families import bambiFamilyName, hasSigmaEps, FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF
from metabeta.utils.io import datasetFilename, runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.padding import padToModel, unpad
from metabeta.utils.preprocessing import rescaleData
from metabeta.utils.sampling import setSeed

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results'

DEFAULT_CONFIGS = ['small-n-sampled']
DEFAULT_SCALES = [0.0, 0.25, 0.5, 1.0, 2.0]

# (display name, extractor, higher_is_better)
METRICS = [
    ('R_ffx', lambda r: r['corr_ffx'], True),
    ('R_σ', lambda r: r['corr_sigma_rfx'], True),
    ('R_rfx', lambda r: r['corr_rfx'], True),
]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Structural misspecification: metabeta vs NUTS.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names')
    parser.add_argument('--scales', nargs='+', type=float, default=DEFAULT_SCALES, help='nonlinearity scales (0 = baseline)')
    parser.add_argument('--max_datasets', type=int, default=16, help='max datasets to evaluate (NUTS is slow)')
    parser.add_argument('--nuts_draws', type=int, default=1000, help='NUTS posterior draws per chain')
    parser.add_argument('--nuts_tune', type=int, default=2000, help='NUTS tuning steps')
    parser.add_argument('--nuts_chains', type=int, default=4, help='NUTS chains')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR), help='output directory')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    return parser.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Setup helpers (shared pattern with other experiments)
# ---------------------------------------------------------------------------


def loadEvalConfig(name: str, **overrides) -> argparse.Namespace:
    path = EVAL_CFG_DIR / f'{name}.yaml'
    assert path.exists(), f'eval config not found: {path}'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['name'] = name
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def initModel(cfg: argparse.Namespace, device: torch.device):
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


def getDataloader(data_cfg: dict, partition: str, batch_size: int | None = None) -> Dataloader:
    data_fname = datasetFilename(data_cfg, partition)
    data_path = METABETA / 'outputs' / 'data' / data_fname
    assert data_path.exists(), f'data not found: {data_path}'
    sortish = batch_size is not None
    return Dataloader(data_path, batch_size=batch_size, sortish=sortish)


def resetRng(model: Approximator, seed: int) -> None:
    model.posterior_g.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore
    model.posterior_l.base_dist.base.rng = np.random.default_rng(seed)  # type: ignore


# ---------------------------------------------------------------------------
# Perturbation: add nonlinear term to y
# ---------------------------------------------------------------------------


def perturbDataset(ds: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    """Add scale × mean-centered(X)² nonlinear term to y.

    For each predictor column (excluding the intercept), compute x_j - mean(x_j),
    square it, and add scale × sum_j(centered_x_j²) to y.  This creates a
    structural misspecification that grows with ``scale``.
    """
    if scale == 0.0:
        return ds
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in ds.items()}
    X = ds['X']  # (n, d) — column 0 is intercept
    d = int(ds['d'])
    if d <= 1:
        return out  # intercept-only model, nothing to perturb
    predictors = X[:, 1:d]  # (n, d-1)
    centered = predictors - predictors.mean(axis=0, keepdims=True)
    nonlinear = (centered ** 2).sum(axis=1)  # (n,)
    out['y'] = ds['y'] + scale * nonlinear
    return out


# ---------------------------------------------------------------------------
# NUTS fitting (lightweight, per-dataset, adapted from simulation/fit.py)
# ---------------------------------------------------------------------------


def pandify(ds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert unpadded dataset to a DataFrame for Bambi."""
    n, d = int(ds['n']), int(ds['d'])
    df = pd.DataFrame(index=range(n))
    df['i'] = ds['groups']
    df['y'] = ds['y']
    for j in range(1, d):
        df[f'x{j}'] = ds['X'][:, j]
    return df


def formulate(ds: dict[str, np.ndarray]) -> str:
    """Build Bambi formula string."""
    d, q = int(ds['d']), int(ds['q'])
    correlated = float(ds.get('eta_rfx', 0)) > 0
    fixed = ' + '.join(f'x{j}' for j in range(1, d))
    slopes = [f'x{j}' for j in range(1, q)]

    if correlated:
        random = '(1 | i)' if not slopes else f"(1 + {' + '.join(slopes)} | i)"
    else:
        random_parts = ['(1 | i)']
        random_parts.extend(f'(0 + {s} | i)' for s in slopes)
        random = ' + '.join(random_parts)

    if fixed:
        return f'y ~ 1 + {fixed} + {random}'
    return f'y ~ 1 + {random}'


def priorize(ds: dict[str, np.ndarray]) -> dict[str, bmb.Prior]:
    """Build Bambi priors from true hyperparameters."""
    d, q = int(ds['d']), int(ds['q'])
    nu_ffx = ds['nu_ffx']
    tau_ffx = ds['tau_ffx']
    tau_rfx = ds['tau_rfx']
    family_ffx = int(ds.get('family_ffx', 0))
    family_sigma_rfx = int(ds.get('family_sigma_rfx', 0))
    priors = {}

    ffx_name = FFX_FAMILIES[family_ffx]
    for j in range(d):
        key = 'Intercept' if j == 0 else f'x{j}'
        if ffx_name == 'normal':
            priors[key] = bmb.Prior('Normal', mu=nu_ffx[j], sigma=tau_ffx[j])
        elif ffx_name == 'student':
            priors[key] = bmb.Prior('StudentT', nu=STUDENT_DF, mu=nu_ffx[j], sigma=tau_ffx[j])

    sigma_name = SIGMA_FAMILIES[family_sigma_rfx]
    for j in range(q):
        key = '1|i' if j == 0 else f'x{j}|i'
        if sigma_name == 'halfnormal':
            sigma = bmb.Prior('HalfNormal', sigma=tau_rfx[j])
        elif sigma_name == 'halfstudent':
            sigma = bmb.Prior('HalfStudentT', nu=STUDENT_DF, sigma=tau_rfx[j])
        elif sigma_name == 'exponential':
            sigma = bmb.Prior('Exponential', lam=1.0 / (tau_rfx[j] + 1e-12))
        else:
            raise ValueError(f'unknown sigma family: {sigma_name}')
        priors[key] = bmb.Prior('Normal', mu=0, sigma=sigma)

    if 'tau_eps' in ds:
        tau_eps = ds['tau_eps']
        family_sigma_eps = int(ds.get('family_sigma_eps', 0))
        eps_name = SIGMA_FAMILIES[family_sigma_eps]
        if eps_name == 'halfnormal':
            priors['sigma'] = bmb.Prior('HalfNormal', sigma=tau_eps)
        elif eps_name == 'halfstudent':
            priors['sigma'] = bmb.Prior('HalfStudentT', nu=STUDENT_DF, sigma=tau_eps)
        elif eps_name == 'exponential':
            priors['sigma'] = bmb.Prior('Exponential', lam=1.0 / (tau_eps + 1e-12))
    return priors


def fitNuts(
    ds: dict[str, np.ndarray],
    draws: int = 1000,
    tune: int = 2000,
    chains: int = 4,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Fit a single dataset with NUTS and return posterior means.

    Returns dict with keys: ffx (d,), sigma_rfx (q,), rfx (m, q).
    """
    likelihood_family = int(ds.get('likelihood_family', 0))
    df = pandify(ds)
    form = formulate(ds)
    priors = priorize(ds) if 'nu_ffx' in ds else None
    family = bambiFamilyName(likelihood_family)
    model = bmb.Model(formula=form, data=df, family=family, categorical='i', priors=priors)
    model.build()

    trace = model.fit(
        tune=tune,
        draws=draws,
        chains=chains,
        cores=1,
        inference_method='pymc',
        random_seed=seed,
        return_inferencedata=True,
    )

    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])

    # extract posterior samples and compute means
    def _extract(name: str) -> np.ndarray:
        x = trace.posterior[name].to_numpy()
        return x.reshape(-1, *x.shape[2:])

    ffx_samples = np.stack([_extract('Intercept' if j == 0 else f'x{j}') for j in range(d)], axis=-1)
    ffx_mean = ffx_samples.mean(axis=0)  # (d,)

    sigma_rfx_samples = np.stack(
        [_extract('1|i_sigma' if j == 0 else f'x{j}|i_sigma') for j in range(q)], axis=-1
    )
    sigma_rfx_mean = sigma_rfx_samples.mean(axis=0)  # (q,)

    rfx_samples = np.stack(
        [_extract('1|i' if j == 0 else f'x{j}|i') for j in range(q)], axis=-1
    )  # (S, m, q)
    rfx_mean = rfx_samples.mean(axis=0)  # (m, q)

    return {
        'ffx': ffx_mean,
        'sigma_rfx': sigma_rfx_mean,
        'rfx': rfx_mean,
    }


# ---------------------------------------------------------------------------
# Metabeta inference (single dataset)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def sampleMetabeta(
    model: Approximator,
    ds: dict[str, np.ndarray],
    cfg: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run metabeta on a single (padded) dataset and return posterior means.

    Returns dict with keys: ffx (d,), sigma_rfx (q,), rfx (m, q).
    """
    d_actual, q_actual, m_actual = int(ds['d']), int(ds['q']), int(ds['m'])

    # padToModel mutates in-place, so copy first to preserve the original
    ds_copy = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in ds.items()}
    padded = padToModel(ds_copy, cfg.max_d, cfg.max_q)
    batch = collateGrouped([padded])
    batch = toDevice(batch, device)

    proposal = model.estimate(batch, n_samples=cfg.n_samples)
    if cfg.rescale:
        proposal.rescale(batch['sd_y'])
    proposal.to('cpu')

    # extract posterior means, trim padding
    # global: (B, S, D) → [0] → (S, D) → mean(0) → (D,)
    ffx_mean = proposal.ffx[0].mean(dim=0)[:d_actual].numpy()
    sigma_rfx_mean = proposal.sigma_rfx[0].mean(dim=0)[:q_actual].numpy()
    # local: (B, m, S, q) → [0] → (m, S, q) → mean(1) → (m, q)
    rfx_mean = proposal.rfx[0].mean(dim=1)[:m_actual, :q_actual].numpy()

    return {
        'ffx': ffx_mean,
        'sigma_rfx': sigma_rfx_mean,
        'rfx': rfx_mean,
    }


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def comparePosteriors(
    mb_list: list[dict[str, np.ndarray]],
    nuts_list: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """Compare metabeta and NUTS posterior means pooled across datasets.

    Concatenates all posterior means, then computes correlation and RMSE
    per parameter type.
    """
    out = {}
    for key in ['ffx', 'sigma_rfx', 'rfx']:
        a = np.concatenate([m[key].ravel() for m in mb_list])
        b = np.concatenate([n[key].ravel() for n in nuts_list])
        if len(a) < 2:
            out[f'corr_{key}'] = np.nan
        else:
            out[f'corr_{key}'] = float(np.corrcoef(a, b)[0, 1])
        out[f'rmse_{key}'] = float(np.sqrt(np.mean((a - b) ** 2)))
    return out


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    configs: list[str],
    scales: list[float],
    max_datasets: int,
    nuts_draws: int,
    nuts_tune: int,
    nuts_chains: int,
) -> list[dict]:
    rows = []

    for config_name in configs:
        print(f'\n{"=" * 60}')
        print(f'Config: {config_name}')
        print(f'{"=" * 60}')

        cfg = loadEvalConfig(config_name, plot=False)
        setSeed(cfg.seed)
        device = setDevice(cfg.device)
        model, data_cfg = initModel(cfg, device)

        # load validation set as raw numpy
        data_fname = datasetFilename(data_cfg, 'valid')
        data_path = METABETA / 'outputs' / 'data' / data_fname
        assert data_path.exists(), f'data not found: {data_path}'
        with np.load(data_path, allow_pickle=True) as raw:
            raw_batch = dict(raw)

        B = len(raw_batch['d'])

        # extract and unpad individual datasets (skip those exceeding model capacity)
        datasets = []
        for i in range(B):
            if len(datasets) >= max_datasets:
                break
            ds = {k: v[i] for k, v in raw_batch.items()}
            sizes = {k: int(ds[k]) for k in ('d', 'q', 'm', 'n')}
            if sizes['d'] > cfg.max_d or sizes['q'] > cfg.max_q:
                continue
            ds = unpad(ds, sizes)
            datasets.append(ds)

        print(f'Using {len(datasets)}/{B} validation datasets (max_d={cfg.max_d}, max_q={cfg.max_q})')

        for scale in scales:
            label = f'λ={scale:g}'
            print(f'\n  --- {label} ---')

            mb_list = []
            nuts_list = []
            for i, ds in enumerate(tqdm(datasets, desc=f'  {label}')):
                perturbed = perturbDataset(ds, scale)

                # metabeta
                setSeed(cfg.seed)
                resetRng(model, cfg.seed)
                mb_list.append(sampleMetabeta(model, perturbed, cfg, device))

                # NUTS
                nuts_list.append(fitNuts(
                    perturbed,
                    draws=nuts_draws,
                    tune=nuts_tune,
                    chains=nuts_chains,
                    seed=cfg.seed,
                ))

            # compare pooled posterior means
            row = {'config': config_name, 'condition': label}
            row.update(comparePosteriors(mb_list, nuts_list))
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def bestIndices(rows: list[dict]) -> dict[str, set[int]]:
    metric_names = [m[0] for m in METRICS]
    direction = {m[0]: m[2] for m in METRICS}
    best: dict[str, set[int]] = {m: set() for m in metric_names}

    configs = []
    seen = set()
    for r in rows:
        if r['config'] not in seen:
            configs.append(r['config'])
            seen.add(r['config'])

    extractors = {m[0]: m[1] for m in METRICS}

    for cfg in configs:
        cfg_rows = [(i, r) for i, r in enumerate(rows) if r['config'] == cfg]
        for metric in metric_names:
            extract = extractors[metric]
            values = []
            for i, r in cfg_rows:
                try:
                    v = extract(r)
                except (KeyError, TypeError):
                    continue
                if v is not None and not np.isnan(v):
                    values.append((i, v))
            if not values:
                continue
            d = direction[metric]
            if d:
                best_idx = max(values, key=lambda x: x[1])[0]
            else:
                best_idx = min(values, key=lambda x: x[1])[0]
            best[metric].add(best_idx)

    return best


def formatTable(rows: list[dict], fmt: str = 'pipe') -> str:
    metric_names = [m[0] for m in METRICS]
    extractors = {m[0]: m[1] for m in METRICS}
    best = bestIndices(rows)

    table_rows = []
    for i, r in enumerate(rows):
        table_row = [r['config'], r['condition']]
        for metric in metric_names:
            try:
                val = extractors[metric](r)
            except (KeyError, TypeError):
                val = None
            if val is None or np.isnan(val):
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
    outdir.mkdir(parents=True, exist_ok=True)

    md_table = formatTable(rows, fmt='pipe')
    md_path = outdir / 'structural.md'
    md_path.write_text(f'# Structural Misspecification Results\n\n{md_table}\n')
    print(f'\nMarkdown saved to {md_path}')

    tex_table = formatTable(rows, fmt='latex')
    tex_path = outdir / 'structural.tex'
    tex_path.write_text(tex_table + '\n')
    print(f'LaTeX saved to {tex_path}')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)

    print(f'Structural misspecification: {len(args.configs)} config(s) × {len(args.scales)} scales')
    print(f'Configs: {args.configs}')
    print(f'Scales: {args.scales}')
    print(f'Max datasets: {args.max_datasets}')
    print(f'NUTS: {args.nuts_draws} draws, {args.nuts_tune} tune, {args.nuts_chains} chains')

    rows = evaluate(
        args.configs,
        args.scales,
        args.max_datasets,
        args.nuts_draws,
        args.nuts_tune,
        args.nuts_chains,
    )

    print(f'\n{formatTable(rows)}')

    save(rows, Path(args.outdir))
    print('\nDone.')
