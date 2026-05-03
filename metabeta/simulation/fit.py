import argparse
from pathlib import Path
import time
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc import adam
import arviz as az
import pytensor

from metabeta.utils.io import datasetFilename
from metabeta.utils.families import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF, hasSigmaEps
from metabeta.utils.padding import aggregate, unpad
from metabeta.utils.templates import setupConfigParser, generateSimulationConfig


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data (template-based, matching generate.py)
    parser.add_argument('--size', type=str, default='small', help='Size preset: tiny|small|medium|large|huge')
    parser.add_argument('--family', type=int, default=0, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type', type=str, default='sampled', help='Dataset type: toy|flat|scm|mixed|sampled|observed')
    parser.add_argument('--config', type=str, help='Path to a saved config.yaml; explicit CLI args override its values')

    parser.add_argument('--idx', type=int, default=0,
        help='Index of dataset in batch to fit (default=0)')
    parser.add_argument('--reintegrate', action='store_true',
        help='Aggregate individual fit files back into the batch (default=False)')
    parser.add_argument('--method', type=str, default='nuts',
        help='Inference method [nuts, advi] (default=nuts)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default=42)')
    parser.add_argument('--tune', type=int, default=2000,
        help='NUTS tuning steps (default=2000)')
    parser.add_argument('--target_accept', type=float, default=0.8,
        help='NUTS target acceptance rate (PyMC default=0.8)')
    parser.add_argument('--max_treedepth', type=int, default=10,
        help='NUTS maximum tree depth (PyMC default=10)')
    parser.add_argument('--draws', type=int, default=1000,
        help='Posterior draws per chain (default=1000)')
    parser.add_argument('--chains', type=int, default=4,
        help='Number of chains (default=4)')
    parser.add_argument('--loop', action='store_true',
        help='Run chains sequentially instead of in parallel (default=False)')
    parser.add_argument('--mp_ctx', type=str, default='forkserver',
        help='Multiprocessing context for parallel chains: fork|forkserver|spawn (default=forkserver)')
    parser.add_argument('--viter', type=int, default=100_000,
        help='ADVI iterations (default=100_000)')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='Adam learning rate for ADVI (default=1e-3)')
    parser.add_argument('--diagonal', action='store_true',
        help='Force diagonal (uncorrelated) RFX covariance even when eta_rfx > 0 (default=False)')
    return setupConfigParser(parser, generateSimulationConfig, 'Fit hierarchical datasets with PyMC.')
# fmt: on


def buildPymc(ds: dict[str, np.ndarray], force_diagonal: bool = False) -> 'pm.Model':
    """Build a PyMC GLMM for a single unpadded dataset.

    Independent (eta_rfx == 0 or q == 1 or force_diagonal): per-dimension
    half-* sigma, non-centered as b_j = z_j * sigma_j.

    Correlated (eta_rfx > 0 and q >= 2 and not force_diagonal): LKJ-Cholesky
    prior on the full covariance, non-centered as b = z @ chol.T.

    All RFX variables are stored as Deterministics named '1|i', 'x1|i',
    '1|i_sigma', … so that extractAll works for both cases.
    """
    d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
    correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2 and not force_diagonal

    y_obs = ds['y'].astype(np.float64)
    X = ds['X'].astype(np.float64).copy()
    Z = X[:, :q].copy()
    groups = ds['groups'].astype(int)

    nu_ffx = ds['nu_ffx'].astype(float)
    tau_ffx = ds['tau_ffx'].astype(float)
    tau_rfx = ds['tau_rfx'].astype(float)

    ffx_family = FFX_FAMILIES[int(ds.get('family_ffx', 0))]
    sigma_family = SIGMA_FAMILIES[int(ds.get('family_sigma_rfx', 0))]
    eps_family = SIGMA_FAMILIES[int(ds.get('family_sigma_eps', 0))]
    likelihood = int(ds.get('likelihood_family', 0))

    def rlabel(j: int, suffix: str = '') -> str:
        return ('1|i' if j == 0 else f'x{j}|i') + suffix

    def half_rv(name: str, family: str, scale, as_dist: bool = False):
        if family == 'halfnormal':
            cls, kw = pm.HalfNormal, {'sigma': scale}
        elif family == 'halfstudent':
            cls, kw = pm.HalfStudentT, {'nu': STUDENT_DF, 'sigma': scale}
        elif family == 'exponential':
            cls, kw = pm.Exponential, {'lam': 1.0 / (np.asarray(scale) + 1e-12)}
        else:
            raise ValueError(f'unsupported sigma family: {family}')
        return cls.dist(**kw) if as_dist else cls(name, **kw)

    with pm.Model() as model:
        betas = []
        for j in range(d):
            name = 'Intercept' if j == 0 else f'x{j}'
            if ffx_family == 'normal':
                betas.append(pm.Normal(name, mu=nu_ffx[j], sigma=tau_ffx[j]))
            elif ffx_family == 'student':
                betas.append(pm.StudentT(name, nu=STUDENT_DF, mu=nu_ffx[j], sigma=tau_ffx[j]))
            else:
                raise ValueError(f'unsupported ffx family: {ffx_family}')

        if correlated:
            chol, _, sigma_vec = pm.LKJCholeskyCov(
                '_lkj_rfx',
                n=q,
                eta=float(ds['eta_rfx']),
                sd_dist=half_rv('', sigma_family, tau_rfx, as_dist=True),
                compute_corr=True,
            )
            for j in range(q):
                pm.Deterministic(rlabel(j, '_sigma'), sigma_vec[j])
            z = pm.Normal('_rfx_offset', 0.0, 1.0, shape=(m, q))
            b = pm.Deterministic('_rfx', pt.dot(z, chol.T))
            for j in range(q):
                pm.Deterministic(rlabel(j), b[:, j])
        else:
            cols = []
            for j in range(q):
                s = half_rv(rlabel(j, '_sigma'), sigma_family, float(tau_rfx[j]))
                z = pm.Normal(rlabel(j, '_offset'), 0.0, 1.0, shape=(m,))
                cols.append(pm.Deterministic(rlabel(j), z * s))
            b = pt.stack(cols, axis=1)

        mu = pt.dot(pt.as_tensor_variable(X), pt.stack(betas))
        mu = mu + (pt.as_tensor_variable(Z) * b[groups]).sum(axis=1)

        if likelihood == 0:
            sigma_eps = half_rv('sigma', eps_family, float(ds.get('tau_eps', 1.0)))
            pm.Normal('y_obs', mu=mu, sigma=sigma_eps, observed=y_obs)
        elif likelihood == 1:
            pm.Bernoulli('y_obs', logit_p=mu, observed=y_obs.astype(int))
        elif likelihood == 2:
            pm.Poisson('y_obs', mu=pt.exp(mu), observed=y_obs.astype(int))
        else:
            raise ValueError(f'unsupported likelihood_family: {likelihood}')

    return model


def extractSingle(trace: 'az.InferenceData', name: str) -> np.ndarray:
    """Flatten (chains, draws, ...) → (1, n_s, ...)."""
    x = trace.posterior[name].to_numpy()  # type: ignore
    x = x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
    return x[None, ...]


def extractAll(
    trace: 'az.InferenceData',
    ds: dict[str, np.ndarray],
    d: int,
    q: int,
    prefix: str,
    force_diagonal: bool = False,
) -> dict[str, np.ndarray]:
    """Extract all posterior arrays from a trace into the (1, n_s, ...) convention."""
    likelihood_family = int(ds.get('likelihood_family', 0))
    ffx = np.concatenate(
        [extractSingle(trace, 'Intercept' if j == 0 else f'x{j}') for j in range(d)], axis=0
    )
    sigma_rfx = np.concatenate(
        [extractSingle(trace, '1|i_sigma' if j == 0 else f'x{j}|i_sigma') for j in range(q)],
        axis=0,
    )
    rfx = np.concatenate(
        [extractSingle(trace, '1|i' if j == 0 else f'x{j}|i') for j in range(q)], axis=0
    ).swapaxes(2, 1)

    out = {f'{prefix}_ffx': ffx, f'{prefix}_sigma_rfx': sigma_rfx, f'{prefix}_rfx': rfx}
    if hasSigmaEps(likelihood_family):
        out[f'{prefix}_sigma_eps'] = extractSingle(trace, 'sigma')
    correlated = float(ds.get('eta_rfx', 0)) > 0 and q >= 2 and not force_diagonal
    if correlated:
        out[f'{prefix}_corr_rfx'] = extractSingle(trace, '_lkj_rfx_corr')
    else:
        n_s = ffx.shape[-1]
        out[f'{prefix}_corr_rfx'] = np.tile(np.eye(q, dtype=ffx.dtype)[None, None], (1, n_s, 1, 1))
    return out


class Fitter:
    def __init__(
        self,
        cfg: argparse.Namespace,
        srcdir: Path = Path(__file__).resolve().parent / '..' / 'outputs' / 'data',
    ) -> None:
        assert cfg.method in ['nuts', 'advi'], 'fit method must be in [nuts, advi]'
        self.cfg = cfg
        self.srcdir = srcdir
        self.outdir = Path(srcdir, self.cfg.data_id, 'fits')
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.fname = datasetFilename(partition='test')
        self.batch_path = Path(self.srcdir, self.cfg.data_id, self.fname)
        assert self.batch_path.exists(), f'{self.batch_path} does not exist'

        with np.load(self.batch_path, allow_pickle=True) as batch:
            self.batch = dict(batch)
        assert 0 <= self.cfg.idx < len(self), 'idx out of bounds'
        self.ds = self._getSingle(self.batch, self.cfg.idx)
        self.outpath = Path(self.outdir, self._outname(cfg.idx))

    def _getSingle(self, batch: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
        ds = {k: v[idx] for k, v in batch.items()}
        sizes = {k: ds[k] for k in list('dqmn')}
        return unpad(ds, sizes)

    def __len__(self) -> int:
        return len(self.batch['y'])

    def _outname(self, idx: int, method: str | None = None) -> str:
        stem = self.batch_path.stem
        use_method = self.cfg.method if method is None else method
        if use_method == 'nuts' and getattr(self.cfg, 'diagonal', False):
            use_method = 'nutsdiag'
        return f'{stem}_{use_method}_{idx:03d}.npz'

    def _buildPymc(self, ds: dict[str, np.ndarray]) -> pm.Model:
        return buildPymc(ds, force_diagonal=getattr(self.cfg, 'diagonal', False))

    def _extract(self, trace: az.InferenceData, name: str) -> np.ndarray:
        return extractSingle(trace, name)

    def _extractAll(
        self, trace: az.InferenceData, d: int, q: int, prefix: str
    ) -> dict[str, np.ndarray]:
        return extractAll(trace, self.ds, d, q, prefix,
                          force_diagonal=getattr(self.cfg, 'diagonal', False))

    def _fitNuts(self, cfg: argparse.Namespace, ds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pymc_model = self._buildPymc(ds)
        t0 = time.perf_counter()
        with pymc_model:
            parallel = not cfg.loop
            trace = pm.sample(
                tune=cfg.tune,
                draws=cfg.draws,
                chains=cfg.chains,
                cores=(cfg.chains if parallel else 1),
                mp_ctx=(cfg.mp_ctx if parallel else None),
                random_seed=cfg.seed,
                target_accept=cfg.target_accept,
                nuts_kwargs={'max_treedepth': cfg.max_treedepth},
                return_inferencedata=True,
                progressbar=True,
            )
        t1 = time.perf_counter()

        d, q = int(ds['d']), int(ds['q'])
        out = self._extractAll(trace, d, q, 'nuts')
        summary = az.summary(trace, kind='diagnostics')
        out['nuts_names'] = summary.index.to_numpy(dtype=str)
        out['nuts_ess'] = summary['ess_bulk'].to_numpy()
        out['nuts_ess_tail'] = summary['ess_tail'].to_numpy()
        out['nuts_rhat'] = summary['r_hat'].to_numpy()
        out['nuts_divergences'] = trace.sample_stats['diverging'].values.sum(-1)  # (chains,)
        out['nuts_draws'] = np.array(cfg.draws)
        tree_depth = trace.sample_stats['tree_depth'].values  # (chains, draws)
        out['nuts_max_treedepth'] = (tree_depth >= cfg.max_treedepth).mean(-1)  # frac saturated per chain
        out['nuts_duration'] = np.array(t1 - t0)
        return out

    def _fitAdvi(
        self,
        cfg: argparse.Namespace,
        ds: dict[str, np.ndarray],
        elbo_every: int = 500,
        es_min_iter: int = 20_000,
        es_window: int = 20,
        es_tol: float = 2e-3,
    ) -> dict[str, np.ndarray]:
        """Fit with ADVI and record the ELBO curve.

        The ELBO is recorded every ``elbo_every`` iterations (default 500).
        PyMC minimises the *negative* ELBO, so ``hist`` contains negative
        values; we negate before storing so the saved array is the ELBO itself
        (should increase / plateau during training).

        Early stopping: once ``es_min_iter`` steps have passed, checks whether
        the last ``es_window`` recorded ELBO values have plateaued —
        mean(|ΔELBO|) / |ELBO| < ``es_tol``. Raises StopIteration (caught by
        PyMC) to terminate cleanly.
        """
        pymc_model = self._buildPymc(ds)
        elbo_steps: list[int] = []
        elbo_vals: list[float] = []

        def _record(approx, hist, i):
            if i % elbo_every == 0:
                elbo_steps.append(i)
                elbo_vals.append(-float(hist[-1]))  # negate loss → ELBO
            # Compare two consecutive smoothed blocks; robust to stochastic noise.
            if i >= es_min_iter and len(elbo_vals) >= 2 * es_window:
                recent = np.mean(elbo_vals[-es_window:])
                prev = np.mean(elbo_vals[-2 * es_window : -es_window])
                delta = abs(recent - prev) / max(abs(recent), 1.0)
                if delta < es_tol:
                    raise StopIteration

        t0 = time.perf_counter()
        with pymc_model:
            mean_field = pm.fit(
                n=cfg.viter,
                method='advi',
                obj_optimizer=adam(learning_rate=cfg.lr),
                callbacks=[_record],
                progressbar=False,
            )
        duration = time.perf_counter() - t0

        trace = mean_field.sample(
            draws=(cfg.draws * cfg.chains),
            random_seed=cfg.seed,
            return_inferencedata=True,
        )

        d, q = int(ds['d']), int(ds['q'])
        out = self._extractAll(trace, d, q, 'advi')
        summary = az.summary(trace, kind='diagnostics')
        out['advi_names'] = summary.index.to_numpy(dtype=str)
        out['advi_ess'] = summary['ess_bulk'].to_numpy()
        out['advi_duration'] = np.array(duration)
        out['advi_elbo'] = np.array(elbo_vals, dtype=np.float64)   # (T,)
        out['advi_elbo_step'] = np.array(elbo_steps, dtype=np.int64)  # (T,)
        out['advi_failed'] = np.array(False)
        return out

    def _adviFailureResult(
        self,
        cfg: argparse.Namespace,
        ds: dict[str, np.ndarray],
        duration: float = np.nan,
    ) -> dict[str, np.ndarray]:
        d, q, m = int(ds['d']), int(ds['q']), int(ds['m'])
        s = int(cfg.draws * cfg.chains)
        out: dict[str, np.ndarray] = {
            'advi_ffx': np.full((d, s), np.nan, dtype=np.float64),
            'advi_sigma_rfx': np.full((q, s), np.nan, dtype=np.float64),
            'advi_rfx': np.full((q, m, s), np.nan, dtype=np.float64),
            'advi_corr_rfx': np.full((1, s, q, q), np.nan, dtype=np.float64),
            'advi_names': np.array([], dtype=str),
            'advi_ess': np.array([], dtype=np.float64),
            'advi_duration': np.array(duration, dtype=np.float64),
            'advi_elbo': np.array([], dtype=np.float64),
            'advi_elbo_step': np.array([], dtype=np.int64),
            'advi_failed': np.array(True),
        }
        if hasSigmaEps(int(ds.get('likelihood_family', 0))):
            out['advi_sigma_eps'] = np.full((1, s), np.nan, dtype=np.float64)
        return out

    def go(self) -> None:
        print(f'Fitting dataset {self.cfg.idx} with {self.cfg.method.upper()}...')
        if self.cfg.method == 'nuts':
            results = self._fitNuts(self.cfg, self.ds)
        elif self.cfg.method == 'advi':
            try:
                results = self._fitAdvi(self.cfg, self.ds)
            except FloatingPointError as exc:
                print(f'ADVI failed for dataset {self.cfg.idx}: {exc}')
                results = self._adviFailureResult(self.cfg, self.ds)
        else:
            raise NotImplementedError
        np.savez_compressed(self.outpath, **results, allow_pickle=True)
        print(f'Saved results to {self.outpath}')

    def _aggregate(self, method: str) -> dict[str, np.ndarray]:
        paths = [Path(self.outdir, self._outname(i, method=method)) for i in range(len(self))]
        for p in paths:
            assert p.exists(), f'cannot aggregate: {p} does not exist'
        fits = []
        for p in paths:
            with np.load(p, allow_pickle=True) as f:
                fit = dict(f)
            if method == 'advi' and 'advi_failed' not in fit:
                fit['advi_failed'] = np.array(False)
            fits.append(fit)
        return aggregate(fits)

    def reintegrate(self) -> None:
        for method in ['nuts', 'advi']:
            self.batch.update(self._aggregate(method))
        fit_suffix = '.fit' + self.batch_path.suffix
        path = self.batch_path.with_suffix(fit_suffix)
        np.savez_compressed(path, **self.batch, allow_pickle=True)
        print(f'Reintegrated NUTS and ADVI fits into {path}')


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'PyTensor tmp directory: {pytensor.config.base_compiledir}')  # type: ignore
    cfg = setup()
    # Provide defaults for fit-specific keys missing when loading from --config YAML
    for _k, _v in [
        ('method', 'nuts'),
        ('idx', 0),
        ('reintegrate', False),
        ('tune', 2000),
        ('target_accept', 0.8),
        ('max_treedepth', 10),
        ('draws', 1000),
        ('chains', 4),
        ('loop', False),
        ('mp_ctx', 'forkserver'),
        ('viter', 100_000),
        ('lr', 1e-3),
        ('diagonal', False),
    ]:
        if not hasattr(cfg, _k):
            setattr(cfg, _k, _v)
    fitter = Fitter(cfg)
    if cfg.reintegrate:
        fitter.reintegrate()
    else:
        fitter.go()
