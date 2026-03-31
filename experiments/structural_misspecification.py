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

NUTS fits are cached per (config, seed, scale, dataset_index, draws, tune, chains)
in ``results/structural_fits/``.  Re-run with ``--refit`` to recompute.

Usage (from experiments/):
    uv run python structural.py
    uv run python structural.py --configs small-n-sampled
    uv run python structural.py --scales 0.0 0.25 0.5 1.0 2.0
    uv run python structural.py --max_datasets 8 --nuts_draws 500
    uv run python structural.py --refit              # force recompute cached fits
    uv run python structural.py --seed 123           # override config seed
"""

import argparse
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm

import bambi as bmb

from metabeta.models.approximator import Approximator
from metabeta.utils.config import (
    assimilateConfig,
    loadDataConfig,
    modelFromYaml,
)
from metabeta.utils.dataloader import collateGrouped, toDevice
from metabeta.utils.families import bambiFamilyName, hasSigmaEps, FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF
from metabeta.utils.io import datasetFilename, runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.padding import padToModel, unpad
from metabeta.posthoc.importance import ImportanceSampler
from metabeta.utils.sampling import setSeed

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results'
FITS_DIR = OUT_DIR / 'structural_fits'

DEFAULT_CONFIGS = ['small-n-mixed', 'mid-n-mixed', 'medium-n-mixed', 'big-n-mixed', 'large-n-mixed']
DEFAULT_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]

# (display name, extractor, higher_is_better)
METRICS = [
    ('R_ffx', lambda r: r['corr_ffx'], True),
    ('R_σ_rfx', lambda r: r['corr_sigma_rfx'], True),
    ('R_σ_eps', lambda r: r.get('corr_sigma_eps', np.nan), True),
    ('R_rfx', lambda r: r['corr_rfx'], True),
]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Structural misspecification: metabeta vs NUTS.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names')
    parser.add_argument('--scales', nargs='+', type=float, default=DEFAULT_SCALES, help='nonlinearity scales (0 = baseline)')
    parser.add_argument('--max_datasets', type=int, default=16, help='max datasets to evaluate (NUTS is slow)')
    parser.add_argument('--seed', type=int, default=None, help='global seed (overrides config seed)')
    parser.add_argument('--nuts_draws', type=int, default=1000, help='NUTS posterior draws per chain')
    parser.add_argument('--nuts_tune', type=int, default=2000, help='NUTS tuning steps')
    parser.add_argument('--nuts_chains', type=int, default=4, help='NUTS chains')
    parser.add_argument('--refit', action='store_true', help='recompute NUTS fits even if cached')
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
# NUTS fitting with caching
# ---------------------------------------------------------------------------


def _fitCachePath(
    data_stem: str, seed: int, scale: float, idx: int,
    draws: int, tune: int, chains: int,
) -> Path:
    """Deterministic cache path for a single NUTS fit.

    Keyed on the data file (not the eval config) so configs that share the
    same validation data reuse cached fits.
    """
    scale_str = f'{scale:.4f}'.replace('.', 'p')
    return FITS_DIR / f'{data_stem}_s{seed}_lam{scale_str}_i{idx:03d}_d{draws}_t{tune}_c{chains}.npz'


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


def _runNuts(
    ds: dict[str, np.ndarray],
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit a single dataset with NUTS and return posterior means."""
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
        cores=chains,
        inference_method='pymc',
        random_seed=seed,
        return_inferencedata=True,
    )

    d, q = int(ds['d']), int(ds['q'])
    likelihood_family = int(ds.get('likelihood_family', 0))

    def _extract(name: str) -> np.ndarray:
        x = trace.posterior[name].to_numpy()
        return x.reshape(-1, *x.shape[2:])

    ffx_mean = np.stack(
        [_extract('Intercept' if j == 0 else f'x{j}') for j in range(d)], axis=-1
    ).mean(axis=0)

    sigma_rfx_mean = np.stack(
        [_extract('1|i_sigma' if j == 0 else f'x{j}|i_sigma') for j in range(q)], axis=-1
    ).mean(axis=0)

    rfx_mean = np.stack(
        [_extract('1|i' if j == 0 else f'x{j}|i') for j in range(q)], axis=-1
    ).mean(axis=0)

    out = {
        'ffx': ffx_mean,
        'sigma_rfx': sigma_rfx_mean,
        'rfx': rfx_mean,
    }
    if hasSigmaEps(likelihood_family):
        out['sigma_eps'] = _extract('sigma').mean(axis=0)
    return out


def fitNuts(
    ds: dict[str, np.ndarray],
    data_stem: str,
    seed: int,
    scale: float,
    idx: int,
    draws: int,
    tune: int,
    chains: int,
    refit: bool,
) -> dict[str, np.ndarray]:
    """Fit NUTS with caching.  Returns dict with ffx, sigma_rfx, rfx means."""
    cache_path = _fitCachePath(data_stem, seed, scale, idx, draws, tune, chains)

    if cache_path.exists() and not refit:
        with np.load(cache_path) as f:
            return dict(f)

    result = _runNuts(ds, draws=draws, tune=tune, chains=chains, seed=seed)

    FITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result


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

    Applies importance sampling when ``cfg.importance`` is true.
    Does NOT rescale — both metabeta and NUTS operate in standardized space.
    """
    d_actual, q_actual, m_actual = int(ds['d']), int(ds['q']), int(ds['m'])
    lf = getattr(cfg, 'likelihood_family', 0)

    # padToModel mutates in-place, so copy first to preserve the original
    ds_copy = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in ds.items()}
    padded = padToModel(ds_copy, cfg.max_d, cfg.max_q)
    batch = collateGrouped([padded])
    batch = toDevice(batch, device)

    proposal = model.estimate(batch, n_samples=cfg.n_samples)

    # importance sampling (in standardized space — no rescale needed)
    if getattr(cfg, 'importance', False):
        imp = ImportanceSampler(batch, sir=False, likelihood_family=lf)
        proposal = imp(proposal)

    proposal.to('cpu')

    # compute (optionally weighted) posterior means
    weights = proposal.weights  # (B, S) or None
    if weights is not None:
        w_g = weights[0].unsqueeze(-1)  # (S, 1) for global params
        w_l = weights[0].unsqueeze(0).unsqueeze(-1)  # (1, S, 1) for local
        ffx_mean = (proposal.ffx[0] * w_g).sum(0)[:d_actual].numpy()
        sigma_rfx_mean = (proposal.sigma_rfx[0] * w_g).sum(0)[:q_actual].numpy()
        rfx_mean = (proposal.rfx[0] * w_l).sum(1)[:m_actual, :q_actual].numpy()
    else:
        ffx_mean = proposal.ffx[0].mean(dim=0)[:d_actual].numpy()
        sigma_rfx_mean = proposal.sigma_rfx[0].mean(dim=0)[:q_actual].numpy()
        rfx_mean = proposal.rfx[0].mean(dim=1)[:m_actual, :q_actual].numpy()

    out = {
        'ffx': ffx_mean,
        'sigma_rfx': sigma_rfx_mean,
        'rfx': rfx_mean,
    }
    if hasSigmaEps(lf):
        if weights is not None:
            out['sigma_eps'] = (proposal.sigma_eps[0] * weights[0]).sum(0).numpy()
        else:
            out['sigma_eps'] = proposal.sigma_eps[0].mean(dim=0).numpy()
    return out


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def comparePosteriors(
    mb_list: list[dict[str, np.ndarray]],
    nuts_list: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """Compare metabeta and NUTS posterior means pooled across datasets.

    Concatenates all posterior means, then computes correlation
    per parameter type.
    """
    out = {}
    for key in ['ffx', 'sigma_rfx', 'sigma_eps', 'rfx']:
        mb_vals = [m[key].ravel() for m in mb_list if key in m]
        nt_vals = [n[key].ravel() for n in nuts_list if key in n]
        if not mb_vals or not nt_vals:
            continue
        a = np.concatenate(mb_vals)
        b = np.concatenate(nt_vals)
        if len(a) < 2:
            out[f'corr_{key}'] = np.nan
        else:
            out[f'corr_{key}'] = float(np.corrcoef(a, b)[0, 1])
    return out


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    configs: list[str],
    scales: list[float],
    max_datasets: int,
    seed_override: int | None,
    nuts_draws: int,
    nuts_tune: int,
    nuts_chains: int,
    refit: bool,
) -> list[dict]:
    rows = []

    for config_name in configs:
        print(f'\n{"=" * 60}')
        print(f'Config: {config_name}')
        print(f'{"=" * 60}')

        cfg = loadEvalConfig(config_name, plot=False)
        seed = seed_override if seed_override is not None else cfg.seed
        setSeed(seed)
        device = setDevice(cfg.device)
        model, data_cfg = initModel(cfg, device)

        # load validation set as raw numpy (use d_tag_valid if available)
        d_tag_valid = getattr(cfg, 'd_tag_valid', cfg.d_tag)
        valid_data_cfg = loadDataConfig(d_tag_valid)
        data_fname = datasetFilename(valid_data_cfg, 'valid')
        data_path = METABETA / 'outputs' / 'data' / data_fname
        assert data_path.exists(), f'data not found: {data_path}'
        with np.load(data_path, allow_pickle=True) as raw:
            raw_batch = dict(raw)

        B = len(raw_batch['d'])

        # deterministic dataset selection
        rng = np.random.default_rng(seed)
        indices = rng.permutation(B)

        datasets = []
        ds_indices = []  # original batch indices, used for cache keys
        for i in indices:
            if len(datasets) >= max_datasets:
                break
            ds = {k: v[i] for k, v in raw_batch.items()}
            sizes = {k: int(ds[k]) for k in ('d', 'q', 'm', 'n')}
            if sizes['d'] > cfg.max_d or sizes['q'] > cfg.max_q:
                continue
            ds = unpad(ds, sizes)
            datasets.append(ds)
            ds_indices.append(int(i))

        print(f'Using {len(datasets)}/{B} validation datasets (max_d={cfg.max_d}, max_q={cfg.max_q})')

        for scale in scales:
            label = f'λ={scale:g}'
            print(f'\n  --- {label} ---')

            mb_list = []
            nuts_list = []
            for j, ds in enumerate(tqdm(datasets, desc=f'  {label}')):
                perturbed = perturbDataset(ds, scale)

                # metabeta
                setSeed(seed)
                resetRng(model, seed)
                mb_list.append(sampleMetabeta(model, perturbed, cfg, device))

                # NUTS (cached by data file, not eval config)
                data_stem = Path(data_fname).stem
                nuts_list.append(fitNuts(
                    perturbed,
                    data_stem=data_stem,
                    seed=seed,
                    scale=scale,
                    idx=ds_indices[j],
                    draws=nuts_draws,
                    tune=nuts_tune,
                    chains=nuts_chains,
                    refit=refit,
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
    print(f'Seed: {args.seed or "from config"}')
    print(f'NUTS: {args.nuts_draws} draws, {args.nuts_tune} tune, {args.nuts_chains} chains')
    print(f'Refit: {args.refit}')

    rows = evaluate(
        args.configs,
        args.scales,
        args.max_datasets,
        args.seed,
        args.nuts_draws,
        args.nuts_tune,
        args.nuts_chains,
        args.refit,
    )

    print(f'\n{formatTable(rows)}')

    save(rows, Path(args.outdir))
    print('\nDone.')
