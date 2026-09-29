"""Reference fits for one dataset: NUTS at three budgets, ADVI, Pathfinder and Laplace.

Every method runs on the PyMC model of ``metabeta.utils.pymc.buildPymc`` and is read back
with ``extractAll``, so the references share priors, parameterisation and output layout.
One process fits one dataset (``--idx``) with one method and writes
``fits/{stem}_{tag}_{idx:03d}.npz`` for each tag the method produces; ``--reintegrate``
aggregates the per-dataset files of a tag into ``{partition}.{tag}.npz``
(``metabeta.utils.fits``). Wall times bracket model construction, compilation and the fit:
the time a user waits for the posterior.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor
from pymc_extras import fit_laplace, fit_pathfinder

from metabeta.utils.evaluation import nutsConverged
from metabeta.utils.fits import saveFits
from metabeta.utils.names import datasetFilename
from metabeta.utils.padding import unpad
from metabeta.utils.pymc import buildPymc, extractAll
from metabeta.utils.templates import generateSimulationConfig, setupConfigParser

# =============================================================================
# Budgets

CHAINS = 4
N_DRAWS = 4000  # posterior draws of every non-NUTS method; NUTS yields CHAINS x draws
# NUTS ladder: level 0 is PyMC's default, the higher levels buy adaptation and precision.
NUTS_LEVELS = (
    dict(tune=1000, draws=1000, target_accept=0.8, max_treedepth=10),
    dict(tune=2000, draws=1000, target_accept=0.9, max_treedepth=10),
    dict(tune=4000, draws=2000, target_accept=0.99, max_treedepth=12),
)
PATHFINDER_PATHS = (4, 20)  # pymc-extras default, then five times as many
# ADVI: one run with PyMC's default optimiser (adagrad_window, Kucukelbir et al. 2017), no
# early stopping; the ELBO is logged at ADVI_ELBO_AT and draws are taken at PyMC's default
# budget (10k iterations, advi0) and ten times that (advi1).
ADVI_ITER = 100_000
ADVI_ELBO_AT = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
ADVI_DRAWS_AT = {10_000: 'advi0', 100_000: 'advi1'}
ELBO_WINDOW = 100  # one-sample ELBO estimates averaged into one logged value
POSTERIOR_KEYS = ('ffx', 'rfx', 'sigma_rfx', 'sigma_eps', 'corr_rfx')  # draw arrays of a fit

_DEFAULT_SRCDIR = Path(__file__).resolve().parent / '..' / 'outputs' / 'data'


def methodTags(method: str, level: int) -> tuple[str, ...]:
    """Fit tags one run of ``method`` at ``level`` writes."""
    levels = {'nuts': len(NUTS_LEVELS), 'pathfinder': len(PATHFINDER_PATHS)}
    if method in levels:
        if not 0 <= level < levels[method]:
            raise ValueError(f'{method} has levels 0..{levels[method] - 1}, got {level}')
        return (f'{method}{level}',)
    if method == 'advi':
        return tuple(ADVI_DRAWS_AT.values())
    if method == 'laplace':
        return ('laplace',)
    raise ValueError(f'unknown method {method!r}; expected nuts, advi, pathfinder or laplace')


def tagMethodLevel(tag: str) -> tuple[str, int]:
    """Inverse of ``methodTags``: the (method, level) run that produces ``tag``."""
    method = tag.rstrip('0123456789')
    if method == 'advi':
        return method, 0  # both ADVI tags come from the same run
    return method, int(tag[len(method) :] or 0)


def aggregateFits(fits: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-dataset fit dicts along a new leading axis.

    Arrays are zero-padded to the largest shape of their key (datasets differ in d, q, m and
    in the number of sampled variables). A key that a failed fit does not carry is NaN
    (numbers) or '' (strings) for that dataset.
    """
    out: dict[str, np.ndarray] = {}
    for key in dict.fromkeys(k for fit in fits for k in fit):
        present = [np.asarray(fit[key]) for fit in fits if key in fit]
        shape = tuple(max(s) for s in zip(*(a.shape for a in present)))
        dtype = np.result_type(*present)
        if len(present) < len(fits) and dtype.kind != 'U':
            dtype = np.dtype(np.float64)
        arr = np.zeros((len(fits), *shape), dtype)
        for i, fit in enumerate(fits):
            if key in fit:
                a = np.asarray(fit[key])
                arr[(i, *(slice(0, s) for s in a.shape))] = a
            elif dtype.kind != 'U':
                arr[i] = np.nan
        out[key] = arr
    return out


def markNonFinite(fits: dict[str, np.ndarray], tag: str) -> dict[str, np.ndarray]:
    """Count a fit whose posterior draws are not finite float32 numbers as failed.

    ``pymc_extras.fit_laplace`` draws NaN without raising when the Hessian at the mode is
    not positive definite, and draws up to 1e154 when the mode of a log-scale lies far in
    the tail; the evaluation stack runs in float32, where both are non-finite. The draws
    of such a dataset are set to NaN like those of a fit that raised, so every consumer
    sees one ``{tag}_failed`` flag.
    """
    keys = [f'{tag}_{k}' for k in POSTERIOR_KEYS if f'{tag}_{k}' in fits]
    failed = fits[f'{tag}_failed'].astype(bool)
    bad = np.zeros_like(failed)
    for key in keys:
        finite = np.isfinite(fits[key]) & (np.abs(fits[key]) <= np.finfo(np.float32).max)
        bad |= ~finite.reshape(len(failed), -1).all(1)
    bad &= ~failed
    for key in keys:
        fits[key][bad] = np.nan
    error = fits[f'{tag}_error'].astype(object)
    error[bad] = 'non-finite draws'
    return {**fits, f'{tag}_failed': failed | bad, f'{tag}_error': error.astype(str)}


def _composite(fit: dict[str, np.ndarray], tag: str, converged: bool, elapsed: float) -> dict:
    """Rename ``{tag}_*`` to ``nuts_*``, thin the draws to N_DRAWS, add level bookkeeping."""
    out = {}
    for key, value in fit.items():
        name = 'nuts_' + key[len(tag) + 1 :]
        if name.endswith('_corr_rfx'):
            step = value.shape[1] // N_DRAWS  # (1, s, q, q)
            value = value[:, ::step]
        elif name.endswith(('_ffx', '_sigma_rfx', '_rfx', '_sigma_eps')):
            step = value.shape[-1] // N_DRAWS  # (..., s)
            value = value[..., ::step]
        out[name] = value
    out['nuts_level'] = np.array(int(tag[len('nuts') :]))
    out['nuts_converged'] = np.array(converged)
    out['nuts_duration'] = np.array(elapsed)
    return out


# =============================================================================
class Fitter:
    def __init__(self, cfg: argparse.Namespace, srcdir: Path = _DEFAULT_SRCDIR) -> None:
        self.cfg = cfg
        self.tags = methodTags(cfg.method, cfg.level)
        fname = datasetFilename(partition=cfg.partition, epoch=getattr(cfg, 'epoch', None) or 1)
        self.batch_path = Path(srcdir, cfg.data_id, fname)
        assert self.batch_path.exists(), f'{self.batch_path} does not exist'
        self.outdir = self.batch_path.parent / 'fits'
        self.outdir.mkdir(parents=True, exist_ok=True)

        with np.load(self.batch_path, allow_pickle=True) as batch:
            self.batch = dict(batch)
        assert 0 <= cfg.idx < len(self), 'idx out of bounds'
        ds = {k: v[cfg.idx] for k, v in self.batch.items()}
        self.ds = unpad(ds, {k: ds[k] for k in 'dqmn'})

    def __len__(self) -> int:
        return len(self.batch['y'])

    def outPath(self, tag: str, idx: int | None = None) -> Path:
        idx = self.cfg.idx if idx is None else idx
        return self.outdir / f'{self.batch_path.stem}_{tag}_{idx:03d}.npz'

    # -------------------------------------------------------------------------
    # Fitting

    def go(self) -> None:
        print(f'Fitting dataset {self.cfg.idx} with {self.cfg.method.upper()} -> {self.tags}')
        fit = {
            'nuts': self._fitNuts,
            'advi': self._fitAdvi,
            'pathfinder': self._fitPathfinder,
            'laplace': self._fitLaplace,
        }[self.cfg.method]
        for tag, arrays in fit().items():
            np.savez_compressed(self.outPath(tag), **arrays)
            state = 'FAILED' if bool(arrays[f'{tag}_failed']) else 'ok'
            print(f'Saved {tag} ({state}) to {self.outPath(tag)}')

    def _model(self) -> pm.Model:
        return buildPymc(self.ds, force_diagonal=self.cfg.diagonal)

    def _extract(self, trace, tag: str, duration: float) -> dict[str, np.ndarray]:
        d, q = int(self.ds['d']), int(self.ds['q'])
        out = extractAll(trace, self.ds, d, q, tag, force_diagonal=self.cfg.diagonal)
        out[f'{tag}_duration'] = np.array(duration)
        out[f'{tag}_failed'] = np.array(False)
        out[f'{tag}_error'] = np.array('')
        return out

    @staticmethod
    def _failure(tag: str, duration: float, exc: Exception) -> dict[str, np.ndarray]:
        # a fit that raises is a recorded outcome of the reference method, not a crash of the
        # array job; aggregateFits fills the missing posterior arrays with NaN
        print(f'{tag} failed: {exc!r}')
        return {
            f'{tag}_duration': np.array(duration),
            f'{tag}_failed': np.array(True),
            f'{tag}_error': np.array(f'{type(exc).__name__}: {exc}'),
        }

    def _fitNuts(self) -> dict[str, dict[str, np.ndarray]]:
        (tag,) = self.tags
        level = NUTS_LEVELS[self.cfg.level]
        t0 = time.perf_counter()
        with self._model():
            trace = pm.sample(
                **level,
                chains=CHAINS,
                cores=CHAINS,
                mp_ctx=self.cfg.mp_ctx,
                random_seed=self.cfg.seed,
                progressbar=False,
            )
        out = self._extract(trace, tag, time.perf_counter() - t0)

        summary = az.summary(trace, kind='diagnostics')
        stats = trace.sample_stats
        energy = stats['energy'].values  # (chains, draws)
        saturated = stats['tree_depth'].values >= level['max_treedepth']  # (chains, draws)
        out.update(
            {
                f'{tag}_names': summary.index.to_numpy(dtype=str),
                f'{tag}_ess': summary['ess_bulk'].to_numpy(),
                f'{tag}_ess_tail': summary['ess_tail'].to_numpy(),
                f'{tag}_rhat': summary['r_hat'].to_numpy(),
                f'{tag}_divergences': stats['diverging'].values.sum(-1),  # (chains,)
                f'{tag}_max_treedepth': saturated.mean(-1),  # fraction at the depth cap
                f'{tag}_n_steps': stats['n_steps'].values.mean(-1),  # leapfrog steps per draw
                f'{tag}_step_size': stats['step_size'].values[:, -1],  # adapted step size
                f'{tag}_accept': stats['acceptance_rate'].values.mean(-1),
                # E-BFMI (Betancourt 2016, arXiv:1604.00695): < 0.3 flags poor energy exploration
                f'{tag}_bfmi': np.square(np.diff(energy, axis=1)).mean(1) / energy.var(1),
                f'{tag}_sampling_time': np.array(trace.posterior.attrs['sampling_time']),
                f'{tag}_level': np.array(self.cfg.level),
                f'{tag}_draws': np.array(level['draws']),
                f'{tag}_tune': np.array(level['tune']),
                f'{tag}_target_accept': np.array(level['target_accept']),
                f'{tag}_chains': np.array(CHAINS),
                f'{tag}_pymc_version': np.array(pm.__version__),
            }
        )
        return {tag: out}

    def _fitAdvi(self) -> dict[str, dict[str, np.ndarray]]:
        elbo: dict[int, float] = {}
        snapshots: dict[str, tuple] = {}  # tag -> (trace, duration)
        sampling = 0.0  # time spent drawing snapshots, excluded from later durations
        t0 = time.perf_counter()

        def record(approx, hist, i):
            nonlocal sampling
            if i in ADVI_ELBO_AT:
                elbo[i] = -float(np.mean(hist[-ELBO_WINDOW:]))  # PyMC minimises -ELBO
            if i in ADVI_DRAWS_AT:
                duration = time.perf_counter() - t0 - sampling
                t_draw = time.perf_counter()
                trace = approx.sample(N_DRAWS, random_seed=self.cfg.seed)
                snapshots[ADVI_DRAWS_AT[i]] = (trace, duration)
                sampling += time.perf_counter() - t_draw

        error = None
        try:
            with self._model():
                pm.fit(
                    n=ADVI_ITER,
                    method='advi',
                    callbacks=[record],
                    random_seed=self.cfg.seed,
                    progressbar=False,
                )
        except Exception as exc:  # e.g. FloatingPointError once the ELBO diverges
            error = exc
        duration = time.perf_counter() - t0 - sampling

        out = {}
        for step, tag in ADVI_DRAWS_AT.items():
            if tag not in snapshots:
                out[tag] = self._failure(tag, duration, error)
                continue
            trace, snap_duration = snapshots[tag]
            arrays = self._extract(trace, tag, snap_duration)
            steps = [s for s in ADVI_ELBO_AT if s <= step]
            arrays[f'{tag}_elbo_step'] = np.array(steps)
            arrays[f'{tag}_elbo'] = np.array([elbo[s] for s in steps])
            out[tag] = arrays
        return out

    def _fitPathfinder(self) -> dict[str, dict[str, np.ndarray]]:
        (tag,) = self.tags
        paths = PATHFINDER_PATHS[self.cfg.level]
        t0 = time.perf_counter()
        try:
            with self._model():
                idata = fit_pathfinder(
                    num_paths=paths,
                    num_draws=N_DRAWS,
                    random_seed=self.cfg.seed,
                    progressbar=False,
                    parallel=False,  # the cluster gives ADVI and Pathfinder one core
                )
        except Exception as exc:
            return {tag: self._failure(tag, time.perf_counter() - t0, exc)}
        out = self._extract(idata, tag, time.perf_counter() - t0)

        pf, lbfgs = idata['pathfinder'], idata['lbfgs']
        out.update(
            {
                f'{tag}_paths': np.array(paths),
                f'{tag}_pareto_k': pf['pareto_k'].values,
                f'{tag}_compile_time': pf['compile_time'].values,
                f'{tag}_compute_time': pf['compute_time'].values,
                f'{tag}_path_status': pf['path_status_counts'].coords['status'].values.astype(str),
                f'{tag}_path_status_counts': pf['path_status_counts'].values,
                f'{tag}_lbfgs_niter': lbfgs['niter'].values,  # (paths,)
                f'{tag}_failed': pf['all_paths_failed'].values,
            }
        )
        return {tag: out}

    def _fitLaplace(self) -> dict[str, dict[str, np.ndarray]]:
        (tag,) = self.tags
        t0 = time.perf_counter()
        try:
            with self._model():
                idata = fit_laplace(draws=N_DRAWS, random_seed=self.cfg.seed, progressbar=False)
        except Exception as exc:
            return {tag: self._failure(tag, time.perf_counter() - t0, exc)}
        out = self._extract(idata, tag, time.perf_counter() - t0)

        opt = idata['optimizer_result']
        out.update(
            {
                f'{tag}_success': opt['success'].values,
                f'{tag}_status': opt['status'].values,
                f'{tag}_iterations': opt['nit'].values,
                f'{tag}_objective': opt['fun'].values,  # negative log posterior at the MAP
                f'{tag}_grad_norm': np.abs(opt['jac'].values).max(),
            }
        )
        return {tag: out}

    # -------------------------------------------------------------------------
    # Reintegration

    def _aggregate(self, tag: str) -> dict[str, np.ndarray]:
        paths = [self.outPath(tag, i) for i in range(len(self))]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)}/{len(paths)} {tag} fits missing, e.g. {missing[0]}'
            )
        fits = []
        for p in paths:
            with np.load(p, allow_pickle=True) as f:
                fits.append(dict(f))
        return markNonFinite(aggregateFits(fits), tag)

    def reintegrate(self, tags: tuple[str, ...] | None = None) -> None:
        """Aggregate the per-dataset files of each tag into ``{partition}.{tag}.npz``."""
        for tag in self.tags if tags is None else tags:
            path = saveFits(self.batch_path, tag, self._aggregate(tag), force=True)
            print(f'Reintegrated {tag} fits into {path}')

    def composeNuts(self) -> None:
        """Write the composite ``nuts`` tag: per dataset the cheapest converged NUTS level.

        The NUTS ladder is run on every dataset; the composite mimics a user who escalates
        the budget until the diagnostics pass (the highest fitted level if none does), so
        ``nuts_duration`` is the cumulative wall time up to the chosen level. Draws are
        thinned to ``N_DRAWS`` so every competitor is evaluated on the same budget.
        """
        levels = [f'nuts{i}' for i in range(len(NUTS_LEVELS))]
        levels = [t for t in levels if all(self.outPath(t, i).exists() for i in range(len(self)))]
        if not levels:
            raise FileNotFoundError(f'no complete NUTS level under {self.outdir}')
        fits = []
        for idx in range(len(self)):
            elapsed = 0.0
            for tag in levels:
                with np.load(self.outPath(tag, idx), allow_pickle=True) as f:
                    fit = dict(f)
                elapsed += float(fit[f'{tag}_duration'])
                converged = bool(nutsConverged({k: v[None] for k, v in fit.items()}, tag)[0])
                if converged or tag == levels[-1]:
                    break
            fits.append(_composite(fit, tag, converged, elapsed))
        path = saveFits(self.batch_path, 'nuts', aggregateFits(fits), force=True)
        n_conv = sum(bool(fit['nuts_converged']) for fit in fits)
        print(f'Composed nuts from {levels} into {path} ({n_conv}/{len(fits)} converged)')


# =============================================================================
# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data (template-based, matching generate.py)
    parser.add_argument('--size', type=str, default='small', help='Size preset: tiny|small|medium|large|huge')
    parser.add_argument('--family', type=int, default=0, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type', type=str, default='sampled', help='Dataset type: toy|flat|scm|mixed|sampled|observed')
    parser.add_argument('--config', type=str, help='Path to a saved config.yaml; explicit CLI args override its values')
    parser.add_argument('--partition', type=str, default='test', choices=['train', 'test', 'valid'])
    parser.add_argument('--epoch', type=int, default=None, help='Epoch number (required when --partition train)')

    # fit
    parser.add_argument('--method', type=str, default='nuts', choices=['nuts', 'advi', 'pathfinder', 'laplace'])
    parser.add_argument('--level', type=int, default=0, help='Budget level: nuts 0|1|2, pathfinder 0|1 (default=0)')
    parser.add_argument('--idx', type=int, default=0, help='Index of the dataset to fit (default=0)')
    parser.add_argument('--reintegrate', action='store_true', help='Aggregate the per-dataset files of this method/level instead of fitting')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mp_ctx', type=str, default='forkserver', help='Multiprocessing context for NUTS chains')
    parser.add_argument('--diagonal', action='store_true', help='Force a diagonal RFX covariance even when eta_rfx > 0')

    cfg = setupConfigParser(parser, generateSimulationConfig, 'Fit one dataset with a PyMC reference method.')
    # keys missing when the config comes from --config YAML
    for key, value in [('partition', 'test'), ('epoch', None), ('method', 'nuts'), ('level', 0), ('idx', 0),
                       ('reintegrate', False), ('seed', 42), ('mp_ctx', 'forkserver'), ('diagonal', False)]:
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    if cfg.partition == 'train' and cfg.epoch is None:
        print('error: --epoch is required when --partition train', file=sys.stderr)
        sys.exit(1)
    return cfg
# fmt: on


if __name__ == '__main__':
    print(f'PyTensor compile directory: {pytensor.config.base_compiledir}')  # type: ignore
    cfg = setup()
    fitter = Fitter(cfg)
    if cfg.reintegrate:
        fitter.reintegrate()
    else:
        fitter.go()
