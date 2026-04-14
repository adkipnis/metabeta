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
from metabeta.utils.families import hasSigmaEps
from metabeta.utils.padding import aggregate, unpad
from metabeta.utils.templates import setupConfigParser, generateSimulationConfig


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data (template-based, matching generate.py)
    parser.add_argument('--size', type=str, default='tiny', help='Size preset: tiny|small|medium|large|huge')
    parser.add_argument('--family', type=int, default=0, help='Likelihood family: 0=normal, 1=bernoulli, 2=poisson')
    parser.add_argument('--ds_type', type=str, default='toy', help='Dataset type: toy|flat|scm|mixed|sampled|observed')
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
    parser.add_argument('--draws', type=int, default=1000,
        help='Posterior draws per chain (default=1000)')
    parser.add_argument('--chains', type=int, default=4,
        help='Number of chains (default=4)')
    parser.add_argument('--loop', action='store_true',
        help='Run chains sequentially instead of in parallel (default=False)')
    parser.add_argument('--viter', type=int, default=50_000,
        help='ADVI iterations (default=50_000)')
    parser.add_argument('--lr', type=float, default=5e-3,
        help='Adam learning rate for ADVI (default=5e-3)')
    return setupConfigParser(parser, generateSimulationConfig, 'Fit hierarchical datasets with PyMC.')
# fmt: on


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
        return f'{stem}_{use_method}_{idx:03d}.npz'

    def _pandify(self, ds: dict[str, np.ndarray]) -> pd.DataFrame:
        """get observations as dataframe"""
        n, d = ds['n'], ds['d']
        df = pd.DataFrame(index=range(n))
        df['i'] = ds['groups']
        df['y'] = ds['y']
        for j in range(1, d):
            df[f'x{j}'] = ds['X'][..., j]
        return df

    def _formulate(self, ds: dict[str, np.ndarray]) -> str:
        """setup bambi model formula based on ds"""
        d, q = ds['d'], ds['q']
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

    def _priorize(self, ds: dict[str, np.ndarray]) -> dict[str, bmb.Prior]:
        """setup bambi priors based on true priors"""
        from metabeta.utils.families import FFX_FAMILIES, SIGMA_FAMILIES, STUDENT_DF

        d, q = ds['d'], ds['q']
        nu_ffx = ds['nu_ffx']
        tau_ffx = ds['tau_ffx']
        tau_rfx = ds['tau_rfx']
        family_ffx = int(ds.get('family_ffx', 0))
        family_sigma_rfx = int(ds.get('family_sigma_rfx', 0))
        priors = {}

        # fixed effects
        if not self.cfg.respecify_ffx:
            ffx_name = FFX_FAMILIES[family_ffx]
            for j in range(d):
                key = 'Intercept' if j == 0 else f'x{j}'
                if ffx_name == 'normal':
                    priors[key] = bmb.Prior('Normal', mu=nu_ffx[j], sigma=tau_ffx[j])
                elif ffx_name == 'student':
                    priors[key] = bmb.Prior(
                        'StudentT', nu=STUDENT_DF, mu=nu_ffx[j], sigma=tau_ffx[j]
                    )

        # random effects variance
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

        # noise variance (normal likelihood only)
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

    def bambify(self, ds: dict[str, np.ndarray]) -> bmb.Model:
        """setup bambi model from dataset dict
        optionally allow bambi to setup its own priors for the fixed effects"""
        likelihood_family = int(ds.get('likelihood_family', 0))
        df = self._pandify(ds)
        form = self._formulate(ds)
        priors = None
        if 'nu_ffx' in ds:
            priors = self._priorize(ds)
        family = bambiFamilyName(likelihood_family)
        model = bmb.Model(
            formula=form,
            data=df,
            family=family,
            categorical='i',
            priors=priors,
        )
        model.build()
        return model

    def _extract(self, trace: az.InferenceData, name: str) -> np.ndarray:
        """extract {name} samples from posterior"""
        x = trace.posterior[name].to_numpy()  # type: ignore
        shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
        x = x.reshape(shape)
        return x[None, ...]

    def _extractAll(
        self, trace: az.InferenceData, d: int, q: int, prefix: str
    ) -> dict[str, np.ndarray]:
        """extract all posterior samples"""
        likelihood_family = int(self.ds.get('likelihood_family', 0))

        # fixed effects
        ffx = []
        for j in range(d):
            key = 'Intercept' if j == 0 else f'x{j}'
            val = self._extract(trace, key)
            ffx.append(val)
        ffx = np.concatenate(ffx, axis=0)

        # variances
        sigma_rfx = []
        for j in range(q):
            key = '1|i_sigma' if j == 0 else f'x{j}|i_sigma'
            val = self._extract(trace, key)
            sigma_rfx.append(val)
        sigma_rfx = np.concatenate(sigma_rfx, axis=0)

        # random effects
        rfx = []
        for j in range(q):
            key = '1|i' if j == 0 else f'x{j}|i'
            val = self._extract(trace, key)
            rfx.append(val)
        rfx = np.concatenate(rfx, axis=0).swapaxes(2, 1)

        # package
        out = {
            f'{prefix}_ffx': ffx,
            f'{prefix}_sigma_rfx': sigma_rfx,
            f'{prefix}_rfx': rfx,
        }
        if hasSigmaEps(likelihood_family):
            out[f'{prefix}_sigma_eps'] = self._extract(trace, 'sigma')
        return out

    def _fitNuts(self, cfg: argparse.Namespace, ds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """fit NUTS-based MCMC model and return samples and diagnostics"""
        model = self.bambify(ds)
        t0 = time.perf_counter()
        trace = model.fit(
            tune=cfg.tune,
            draws=cfg.draws,
            chains=cfg.chains,
            cores=(1 if cfg.loop else cfg.chains),
            inference_method='pymc',
            random_seed=cfg.seed,
            return_inferencedata=True,
        )
        t1 = time.perf_counter()

        # --- extract samples
        d, q = int(ds['d']), int(ds['q'])
        out = self._extractAll(trace, d, q, 'nuts')

        # --- extract diagnostics
        summary = az.summary(trace, kind='diagnostics')
        out['nuts_names'] = summary.index.to_numpy(dtype=str)
        out['nuts_ess'] = summary['ess_bulk'].to_numpy()
        out['nuts_divergences'] = trace.sample_stats['diverging'].values.sum(-1)
        out['nuts_duration'] = np.array(t1 - t0)
        return out

    def _fitAdvi(self, cfg: argparse.Namespace, ds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """fit ADVI model (using ADAM) and return samples and diagnostics"""
        model = self.bambify(ds)
        t0 = time.perf_counter()
        mean_field = model.fit(
            inference_method='vi',
            n=cfg.viter,
            obj_optimizer=adam(learning_rate=cfg.lr),
        )
        trace = mean_field.sample(
            draws=(cfg.draws * cfg.chains),
            random_seed=cfg.seed,
            return_inferencedata=True,
        )
        t1 = time.perf_counter()

        # --- extract samples
        d, q = int(ds['d']), int(ds['q'])
        out = self._extractAll(trace, d, q, 'advi')

        # --- extract diagnostics
        summary = az.summary(trace, kind='diagnostics')
        out['advi_names'] = summary.index.to_numpy(dtype=str)
        out['advi_ess'] = summary['ess_bulk'].to_numpy()
        out['advi_duration'] = np.array(t1 - t0)
        return out

    def go(self) -> None:
        """wrapper for fitting a single dataset with index {idx} in cfg"""
        print(f'Fitting dataset {self.cfg.idx} with {self.cfg.method.upper()}...')
        if self.cfg.method == 'nuts':
            results = self._fitNuts(self.cfg, self.ds)
        elif self.cfg.method == 'advi':
            results = self._fitAdvi(self.cfg, self.ds)
        else:
            raise NotImplementedError
        np.savez_compressed(self.outpath, **results, allow_pickle=True)
        print(f'Saved results to {self.outpath}')

    def _aggregate(self, method: str) -> dict[str, np.ndarray]:
        """load all fits of {method}, aggregate and update batch"""
        # get paths
        paths = [Path(self.outdir, self._outname(i, method=method)) for i in range(len(self))]

        # check if all files exist
        for p in paths:
            assert p.exists(), f'cannot aggregate: the fit file {p} does not exist, yet.'

        # load fits safely
        fits = []
        for p in paths:
            with np.load(p, allow_pickle=True) as f:
                fits.append(dict(f))
        return aggregate(fits)

    def reintegrate(self) -> None:
        # update batch for each fit method
        for method in ['nuts', 'advi']:
            fits = self._aggregate(method)
            self.batch.update(fits)

        # save updated batch
        fit_suffix = '.fit' + self.batch_path.suffix
        path = self.batch_path.with_suffix(fit_suffix)
        np.savez_compressed(path, **self.batch, allow_pickle=True)
        print(f'Reintegrated NUTS and ADVI fits into {path}')


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'PyTensor tmp directory: {pytensor.config.base_compiledir}')  # type: ignore
    cfg = setup()
    fitter = Fitter(cfg)
    if cfg.reintegrate:
        fitter.reintegrate()
    else:
        fitter.go()
