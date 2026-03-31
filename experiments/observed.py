"""
Out-of-distribution evaluation: metabeta vs NUTS/ADVI on real observed data.

Subsample real datasets from the preprocessed test pool (no parameter
simulation — y is kept as-is) and compare metabeta posteriors against
NUTS and ADVI.  Since there are no ground-truth parameters, comparison
is based on:
    - Predictive NLL (log p(y | posterior samples))
    - Correlation of posterior means between methods

NUTS and ADVI fits are cached per (config, seed, source, dataset_index, ...)
in ``results/observed_fits/``.  Re-run with ``--refit`` to recompute.

Workflow:
    1. Load model checkpoint and evaluation config
    2. Subsample observed datasets via Subsampler
    3. For each dataset:
        a. Run metabeta inference
        b. Run NUTS (per-dataset, cached)
        c. Run ADVI (per-dataset, cached)
    4. Compute predictive NLL and parameter correlations
    5. Save results table

Usage (from experiments/):
    uv run python observed.py
    uv run python observed.py --configs small-n-sampled
    uv run python observed.py --n_datasets 8 --nuts_draws 500
    uv run python observed.py --source math__grp_group
    uv run python observed.py --refit                       # force recompute cached fits
"""

import argparse
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymc import adam
from tabulate import tabulate
from tqdm import tqdm

import bambi as bmb
import arviz as az

from metabeta.models.approximator import Approximator
from metabeta.simulation.emulator import Subsampler
from metabeta.simulation.prior import bambiDefaultPriors
from metabeta.utils.config import (
    assimilateConfig,
    loadDataConfig,
    modelFromYaml,
)
from metabeta.utils.dataloader import collateGrouped, toDevice
from metabeta.utils.families import (
    bambiFamilyName,
    hasSigmaEps,
    logLikelihood,
    FFX_FAMILIES,
    SIGMA_FAMILIES,
    STUDENT_DF,
)
from metabeta.utils.io import runName, setDevice
from metabeta.utils.logger import setupLogging
from metabeta.utils.padding import padToModel
from metabeta.utils.sampling import setSeed, truncLogUni

DIR = Path(__file__).resolve().parent
METABETA = DIR / '..' / 'metabeta'
EVAL_CFG_DIR = METABETA / 'evaluation' / 'configs'
OUT_DIR = DIR / 'results'
FITS_DIR = OUT_DIR / 'observed_fits'

DEFAULT_CONFIGS = ['small-n-sampled']

METRICS = [
    ('NLL_mb', lambda r: r['nll_metabeta'], False),
    ('NLL_nuts', lambda r: r['nll_nuts'], False),
    ('NLL_advi', lambda r: r['nll_advi'], False),
    ('R_ffx', lambda r: r['corr_ffx_mb_nuts'], True),
    ('R_σ_rfx', lambda r: r['corr_sigma_rfx_mb_nuts'], True),
    ('R_σ_eps', lambda r: r.get('corr_sigma_eps_mb_nuts', np.nan), True),
    ('R_rfx', lambda r: r['corr_rfx_mb_nuts'], True),
]


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Out-of-distribution: metabeta vs NUTS/ADVI on observed data.')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='evaluation config names')
    parser.add_argument('--source', type=str, default='all', help='source dataset name or "all"')
    parser.add_argument('--n_datasets', type=int, default=16, help='number of subsampled datasets')
    parser.add_argument('--nuts_draws', type=int, default=1000, help='NUTS posterior draws per chain')
    parser.add_argument('--nuts_tune', type=int, default=2000, help='NUTS tuning steps')
    parser.add_argument('--nuts_chains', type=int, default=4, help='NUTS chains')
    parser.add_argument('--advi_iter', type=int, default=50_000, help='ADVI iterations')
    parser.add_argument('--advi_lr', type=float, default=5e-3, help='ADVI learning rate')
    parser.add_argument('--refit', action='store_true', help='recompute NUTS/ADVI fits even if cached')
    parser.add_argument('--outdir', type=str, default=str(OUT_DIR), help='output directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--verbosity', type=int, default=1, help='0=warnings | 1=info | 2=debug')
    return parser.parse_args()
# fmt: on


# ---------------------------------------------------------------------------
# Setup helpers
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
# Subsample observed datasets
# ---------------------------------------------------------------------------


def subsampleDatasets(
    n_datasets: int,
    source: str,
    likelihood_family: int,
    max_d: int,
    max_q: int,
    min_m: int,
    max_m: int,
    min_n: int,
    max_n: int,
    max_n_total: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Generate n_datasets subsampled observed datasets using Subsampler."""
    rng = np.random.default_rng(seed)
    datasets = []

    for _ in range(n_datasets):
        # sample dimensions (mirroring Generator._genSizes logic)
        d = int(truncLogUni(rng, low=2, high=max_d + 1, size=1, round=True)[0])
        q = int(truncLogUni(rng, low=1, high=max_q + 1, size=1, round=True)[0])
        q = min(d, q)
        m = int(truncLogUni(rng, low=min_m, high=max_m + 1, size=1, round=True)[0])
        ns = truncLogUni(rng, low=min_n, high=max_n + 1, size=m, round=True).astype(int)

        # cap total n
        total = int(ns.sum())
        if total > max_n_total:
            factor = max_n_total / total
            ns = np.maximum(min_n, np.floor(ns * factor).astype(int))

        subsampler = Subsampler(
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            source=source,
            likelihood_family=likelihood_family,
            min_m=min_m,
            min_n=min_n,
            max_n=max_n,
        )
        obs = subsampler.sample(d, ns)

        # build bambi-default hyperparameters
        hyperparams = bambiDefaultPriors(d, q, likelihood_family)

        # assemble dataset with NaN placeholders for ground-truth params
        m_actual = len(obs['ns'])
        n_actual = len(obs['y'])
        ds = {
            'ffx': np.full(d, np.nan),
            'sigma_rfx': np.full(q, np.nan),
            'corr_rfx': np.full((q, q), np.nan),
            'rfx': np.full((m_actual, q), np.nan),
            **hyperparams,
            'y': obs['y'],
            'X': obs['X'],
            'groups': obs['groups'],
            'm': np.array(m_actual),
            'n': np.array(n_actual),
            'ns': obs['ns'],
            'd': np.array(d),
            'q': np.array(q),
            'sd_y': obs['sd_y'],
        }
        if hasSigmaEps(likelihood_family):
            ds['sigma_eps'] = np.array(np.nan)
            ds['r_squared'] = np.array(np.nan)

        datasets.append(ds)

    return datasets


# ---------------------------------------------------------------------------
# Bambi helpers (NUTS and ADVI)
# ---------------------------------------------------------------------------


def pandify(ds: dict[str, np.ndarray]) -> pd.DataFrame:
    n, d = int(ds['n']), int(ds['d'])
    df = pd.DataFrame(index=range(n))
    df['i'] = ds['groups']
    df['y'] = ds['y']
    for j in range(1, d):
        df[f'x{j}'] = ds['X'][:, j]
    return df


def formulate(ds: dict[str, np.ndarray]) -> str:
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
    """Build Bambi priors from bambi-default hyperparameters."""
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


def _extractSamples(
    trace: az.InferenceData,
    d: int,
    q: int,
    m: int,
    has_sigma_eps: bool,
) -> dict[str, np.ndarray]:
    """Extract posterior samples from an arviz trace."""

    def _get(name: str) -> np.ndarray:
        x = trace.posterior[name].to_numpy()
        return x.reshape(-1, *x.shape[2:])

    ffx = np.stack([_get('Intercept' if j == 0 else f'x{j}') for j in range(d)], axis=-1)
    sigma_rfx = np.stack(
        [_get('1|i_sigma' if j == 0 else f'x{j}|i_sigma') for j in range(q)], axis=-1
    )
    rfx = np.stack([_get('1|i' if j == 0 else f'x{j}|i') for j in range(q)], axis=-1)  # (S, m, q)

    out = {
        'ffx': ffx,  # (S, d)
        'sigma_rfx': sigma_rfx,  # (S, q)
        'rfx': rfx,  # (S, m, q)
    }
    if has_sigma_eps:
        out['sigma_eps'] = _get('sigma')  # (S,)
    return out


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _nutsCachePath(
    config: str,
    seed: int,
    source: str,
    idx: int,
    draws: int,
    tune: int,
    chains: int,
) -> Path:
    return FITS_DIR / f'{config}_s{seed}_{source}_i{idx:03d}_nuts_d{draws}_t{tune}_c{chains}.npz'


def _adviCachePath(
    config: str,
    seed: int,
    source: str,
    idx: int,
    n_iter: int,
    draws: int,
) -> Path:
    return FITS_DIR / f'{config}_s{seed}_{source}_i{idx:03d}_advi_n{n_iter}_d{draws}.npz'


def _runNuts(
    ds: dict[str, np.ndarray],
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> dict[str, np.ndarray]:
    likelihood_family = int(ds.get('likelihood_family', 0))
    df = pandify(ds)
    form = formulate(ds)
    priors = priorize(ds)
    family = bambiFamilyName(likelihood_family)
    model = bmb.Model(formula=form, data=df, family=family, categorical='i', priors=priors)
    model.build()

    trace = model.fit(
        tune=tune,
        draws=draws,
        chains=chains,
        inference_method='pymc',
        random_seed=seed,
        return_inferencedata=True,
    )

    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    return _extractSamples(trace, d, q, m, hasSigmaEps(likelihood_family))


def _runAdvi(
    ds: dict[str, np.ndarray],
    n_iter: int,
    lr: float,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    likelihood_family = int(ds.get('likelihood_family', 0))
    df = pandify(ds)
    form = formulate(ds)
    priors = priorize(ds)
    family = bambiFamilyName(likelihood_family)
    model = bmb.Model(formula=form, data=df, family=family, categorical='i', priors=priors)
    model.build()

    mean_field = model.fit(
        inference_method='vi',
        n=n_iter,
        obj_optimizer=adam(learning_rate=lr),
    )
    trace = mean_field.sample(
        draws=draws,
        random_seed=seed,
        return_inferencedata=True,
    )

    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    return _extractSamples(trace, d, q, m, hasSigmaEps(likelihood_family))


def fitNuts(
    ds: dict[str, np.ndarray],
    config: str,
    seed: int,
    source: str,
    idx: int,
    draws: int,
    tune: int,
    chains: int,
    refit: bool,
) -> dict[str, np.ndarray]:
    """Fit NUTS with caching.  Returns posterior samples."""
    cache_path = _nutsCachePath(config, seed, source, idx, draws, tune, chains)

    if cache_path.exists() and not refit:
        with np.load(cache_path) as f:
            return dict(f)

    result = _runNuts(ds, draws=draws, tune=tune, chains=chains, seed=seed)

    FITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result


def fitAdvi(
    ds: dict[str, np.ndarray],
    config: str,
    seed: int,
    source: str,
    idx: int,
    n_iter: int,
    lr: float,
    draws: int,
    refit: bool,
) -> dict[str, np.ndarray]:
    """Fit ADVI with caching.  Returns posterior samples."""
    cache_path = _adviCachePath(config, seed, source, idx, n_iter, draws)

    if cache_path.exists() and not refit:
        with np.load(cache_path) as f:
            return dict(f)

    result = _runAdvi(ds, n_iter=n_iter, lr=lr, draws=draws, seed=seed)

    FITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result


# ---------------------------------------------------------------------------
# Metabeta inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def sampleMetabeta(
    model: Approximator,
    ds: dict[str, np.ndarray],
    cfg: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run metabeta on a single dataset and return posterior samples."""
    d_actual, q_actual, m_actual = int(ds['d']), int(ds['q']), int(ds['m'])
    likelihood_family = int(ds.get('likelihood_family', 0))

    ds_copy = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in ds.items()}
    padded = padToModel(ds_copy, cfg.max_d, cfg.max_q)
    batch = collateGrouped([padded])
    batch = toDevice(batch, device)

    proposal = model.estimate(batch, n_samples=cfg.n_samples)
    if cfg.rescale:
        proposal.rescale(batch['sd_y'])
    proposal.to('cpu')

    # global: (B, S, D) → [0] → (S, D)
    ffx = proposal.ffx[0][:, :d_actual].numpy()           # (S, d)
    sigma_rfx = proposal.sigma_rfx[0][:, :q_actual].numpy()  # (S, q)
    # local: (B, m, S, q) → [0] → (m, S, q) → permute → (S, m, q)
    rfx = proposal.rfx[0][:m_actual, :, :q_actual].permute(1, 0, 2).numpy()  # (S, m, q)

    out = {
        'ffx': ffx,
        'sigma_rfx': sigma_rfx,
        'rfx': rfx,
    }
    if hasSigmaEps(likelihood_family):
        out['sigma_eps'] = proposal.sigma_eps[0].numpy()  # (S,)
    return out


# ---------------------------------------------------------------------------
# Predictive NLL
# ---------------------------------------------------------------------------


def predictiveNLL(
    samples: dict[str, np.ndarray],
    ds: dict[str, np.ndarray],
) -> float:
    """Compute average predictive NLL using posterior samples.

    Uses the log-sum-exp trick to estimate log p(y) = log (1/S sum_s p(y|theta_s)).
    Returns per-observation NLL (lower is better).
    """
    likelihood_family = int(ds.get('likelihood_family', 0))
    has_eps = hasSigmaEps(likelihood_family)
    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    n = int(ds['n'])

    ffx_s = torch.from_numpy(samples['ffx']).float()          # (S, d)
    sigma_rfx_s = torch.from_numpy(samples['sigma_rfx']).float()
    rfx_s = torch.from_numpy(samples['rfx']).float()           # (S, m, q)
    S = ffx_s.shape[0]

    if has_eps:
        sigma_eps_s = torch.from_numpy(samples['sigma_eps']).float()  # (S,)
    else:
        sigma_eps_s = torch.zeros(S)

    # prepare observations: need (1, m, n_max, d) grouped structure
    X = torch.from_numpy(ds['X']).float()  # (n, d)
    y = torch.from_numpy(ds['y']).float()  # (n,)
    groups = ds['groups']
    ns = ds['ns']

    # group into (m, n_max, ...) with masking
    n_max = int(ns.max())
    X_grouped = torch.zeros(m, n_max, d)
    y_grouped = torch.zeros(m, n_max)
    mask = torch.zeros(m, n_max)

    for g in range(m):
        idx = np.where(groups == g)[0]
        ng = len(idx)
        X_grouped[g, :ng] = X[idx]
        y_grouped[g, :ng] = y[idx]
        mask[g, :ng] = 1.0

    # expand to batch dim: (1, ...)
    X_b = X_grouped.unsqueeze(0)           # (1, m, n_max, d)
    Z_b = X_b[..., :q]                     # (1, m, n_max, q)
    y_b = y_grouped.unsqueeze(0).unsqueeze(-1)  # (1, m, n_max, 1)
    mask_b = mask.unsqueeze(0).unsqueeze(-1)     # (1, m, n_max, 1)

    # samples: (1, S, ...)
    ffx_b = ffx_s.unsqueeze(0)             # (1, S, d)
    sigma_eps_b = sigma_eps_s.unsqueeze(0)  # (1, S)
    rfx_b = rfx_s.permute(1, 0, 2).unsqueeze(0)  # (1, m, S, q)

    # log p(y | theta_s) for each sample — sum over (m, n) dims → (1, S)
    ll = logLikelihood(
        ffx_b,
        sigma_eps_b,
        rfx_b,
        y_b,
        X_b,
        Z_b,
        mask=mask_b,
        likelihood_family=likelihood_family,
    )  # (1, S)

    # log mean exp: log (1/S sum exp(ll_s)) = logsumexp(ll) - log(S)
    log_pred = torch.logsumexp(ll[0], dim=0) - np.log(S)

    # per-observation NLL
    return -float(log_pred) / n


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _posteriorMeans(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce posterior samples to means (one value per parameter)."""
    out = {}
    for key in ('ffx', 'sigma_rfx', 'rfx'):
        out[key] = samples[key].mean(axis=0).ravel()
    if 'sigma_eps' in samples:
        out['sigma_eps'] = np.atleast_1d(samples['sigma_eps'].mean(axis=0))
    return out


def comparePosteriors(
    mb_list: list[dict[str, np.ndarray]],
    nuts_list: list[dict[str, np.ndarray]],
    advi_list: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    """Correlations of posterior means pooled across datasets.

    Each entry in the lists is the output of ``_posteriorMeans`` for one
    dataset.  By concatenating across datasets we get enough points to
    correlate even scalar parameters (sigma_rfx with q=1, sigma_eps).
    """
    keys = ['ffx', 'sigma_rfx', 'rfx']
    if 'sigma_eps' in mb_list[0]:
        keys.append('sigma_eps')

    out = {}
    for key in keys:
        a_mb = np.concatenate([m[key] for m in mb_list])
        a_nuts = np.concatenate([n[key] for n in nuts_list])
        a_advi = np.concatenate([a[key] for a in advi_list])

        if len(a_mb) >= 2:
            out[f'corr_{key}_mb_nuts'] = float(np.corrcoef(a_mb, a_nuts)[0, 1])
            out[f'corr_{key}_mb_advi'] = float(np.corrcoef(a_mb, a_advi)[0, 1])
            out[f'corr_{key}_nuts_advi'] = float(np.corrcoef(a_nuts, a_advi)[0, 1])
        else:
            out[f'corr_{key}_mb_nuts'] = np.nan
            out[f'corr_{key}_mb_advi'] = np.nan
            out[f'corr_{key}_nuts_advi'] = np.nan
    return out


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> list[dict]:
    rows = []

    for config_name in args.configs:
        print(f'\n{"=" * 60}')
        print(f'Config: {config_name}')
        print(f'{"=" * 60}')

        cfg = loadEvalConfig(config_name, plot=False)
        setSeed(cfg.seed)
        device = setDevice(cfg.device)
        model, data_cfg = initModel(cfg, device)

        likelihood_family = getattr(cfg, 'likelihood_family', 0)

        # subsample observed datasets
        print(f'Subsampling {args.n_datasets} observed datasets...')
        datasets = subsampleDatasets(
            n_datasets=args.n_datasets,
            source=args.source,
            likelihood_family=likelihood_family,
            max_d=cfg.max_d,
            max_q=cfg.max_q,
            min_m=data_cfg.get('min_m', 3),
            max_m=data_cfg.get('max_m', 25),
            min_n=data_cfg.get('min_n', 4),
            max_n=data_cfg.get('max_n', 30),
            max_n_total=data_cfg.get('max_n_total', 800),
            seed=args.seed,
        )
        print(f'Generated {len(datasets)} datasets')

        nll_mb_all, nll_nuts_all, nll_advi_all = [], [], []
        mb_means_all, nuts_means_all, advi_means_all = [], [], []

        for i, ds in enumerate(tqdm(datasets, desc='datasets')):
            d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
            n = int(ds['n'])
            tqdm.write(f'  [{i}] d={d} q={q} m={m} n={n}')

            # metabeta
            setSeed(cfg.seed)
            resetRng(model, cfg.seed)
            mb_samples = sampleMetabeta(model, ds, cfg, device)

            # NUTS (cached)
            nuts_samples = fitNuts(
                ds,
                config=config_name,
                seed=args.seed,
                source=args.source,
                idx=i,
                draws=args.nuts_draws,
                tune=args.nuts_tune,
                chains=args.nuts_chains,
                refit=args.refit,
            )

            # ADVI (cached)
            advi_draws = args.nuts_draws * args.nuts_chains
            advi_samples = fitAdvi(
                ds,
                config=config_name,
                seed=args.seed,
                source=args.source,
                idx=i,
                n_iter=args.advi_iter,
                lr=args.advi_lr,
                draws=advi_draws,
                refit=args.refit,
            )

            # predictive NLL
            nll_mb = predictiveNLL(mb_samples, ds)
            nll_nuts = predictiveNLL(nuts_samples, ds)
            nll_advi = predictiveNLL(advi_samples, ds)
            nll_mb_all.append(nll_mb)
            nll_nuts_all.append(nll_nuts)
            nll_advi_all.append(nll_advi)

            tqdm.write(f'    NLL: mb={nll_mb:.3f}  nuts={nll_nuts:.3f}  advi={nll_advi:.3f}')

            # collect posterior means for pooled correlation
            mb_means_all.append(_posteriorMeans(mb_samples))
            nuts_means_all.append(_posteriorMeans(nuts_samples))
            advi_means_all.append(_posteriorMeans(advi_samples))

        # pooled correlations across all datasets
        corrs = comparePosteriors(mb_means_all, nuts_means_all, advi_means_all)

        row = {
            'config': config_name,
            'n_datasets': len(datasets),
            'nll_metabeta': float(np.mean(nll_mb_all)),
            'nll_nuts': float(np.mean(nll_nuts_all)),
            'nll_advi': float(np.mean(nll_advi_all)),
        }
        row.update(corrs)
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def formatTable(rows: list[dict], fmt: str = 'pipe') -> str:
    metric_names = [m[0] for m in METRICS]
    extractors = {m[0]: m[1] for m in METRICS}

    table_rows = []
    for r in rows:
        table_row = [r['config'], r['n_datasets']]
        for metric in metric_names:
            try:
                val = extractors[metric](r)
            except (KeyError, TypeError):
                val = None
            if val is None or np.isnan(val):
                cell = '—'
            else:
                cell = f'{val:.4f}'
            table_row.append(cell)
        table_rows.append(table_row)

    headers = ['Config', 'N'] + metric_names
    tablefmt = 'latex_booktabs' if fmt == 'latex' else 'pipe'
    return tabulate(table_rows, headers=headers, tablefmt=tablefmt, stralign='right')


def save(rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    md_table = formatTable(rows, fmt='pipe')
    md_path = outdir / 'observed.md'
    md_path.write_text(f'# Out-of-Distribution (Observed Data) Results\n\n{md_table}\n')
    print(f'\nMarkdown saved to {md_path}')

    tex_table = formatTable(rows, fmt='latex')
    tex_path = outdir / 'observed.tex'
    tex_path.write_text(tex_table + '\n')
    print(f'LaTeX saved to {tex_path}')

    # also save raw results
    df = pd.DataFrame(rows)
    csv_path = outdir / 'observed.csv'
    df.to_csv(csv_path, index=False)
    print(f'CSV saved to {csv_path}')


if __name__ == '__main__':
    args = setup()
    setupLogging(args.verbosity)

    print(f'Out-of-distribution evaluation: {len(args.configs)} config(s)')
    print(f'Configs: {args.configs}')
    print(f'Source: {args.source}')
    print(f'Datasets: {args.n_datasets}')
    print(f'NUTS: {args.nuts_draws} draws, {args.nuts_tune} tune, {args.nuts_chains} chains')
    print(f'ADVI: {args.advi_iter} iterations, lr={args.advi_lr}')
    print(f'Refit: {args.refit}')

    rows = evaluate(args)

    print(f'\n{formatTable(rows)}')

    save(rows, Path(args.outdir))
    print('\nDone.')
