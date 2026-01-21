import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from pymc import adam
import bambi as bmb
import arviz as az

from metabeta.utils.io import datasetFilename


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fit hierarchical datasets with Bambi.')
    # data dimensions
    parser.add_argument('-d', '--max_d', type=int, default=3, help='Maximum number of fixed effects (intercept + slopes) to draw per linear model (default = 16).')
    parser.add_argument('-q', '--max_q', type=int, default=1, help='Maximum number of random effects (intercept + slopes) to draw per linear model (default = 4).')
    parser.add_argument('--min_m', type=int, default=5, help='Minimum number of groups (default = 5).')
    parser.add_argument('--max_m', type=int, default=30, help='Maximum number of groups (default = 30).')
    parser.add_argument('--min_n', type=int, default=10, help='Minimum number of samples per group (default = 10).')
    parser.add_argument('--max_n', type=int, default=70, help='Maximum number of samples per group (default = 70).')
    # partitions and sources
    parser.add_argument('--type', type=str, default='toy', help='Type of predictors [toy, flat, scm, sampled], (default = toy)')
    parser.add_argument('--source', type=str, default='all', help='Source dataset if type==sampled (default = all)')
    # bambi
    parser.add_argument('--respecify_ffx', action='store_true', help='Use automatic fixed effects priors by bambi instead of known (default = False)')
    parser.add_argument('--method', type=str, default='advi', help='Inference method for bambi [nuts, advi], (default = nuts)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for bambi (default = 42)')
    parser.add_argument('--tune', type=int, default=2000, help='Number of tuning steps (burnin) for MCMC (default = 2000)')
    parser.add_argument('--draws', type=int, default=1000, help='Number of posterior samples (default = 1000)')
    parser.add_argument('--chains', type=int, default=4, help='Number of posterior sampling chains (default = 4)')
    parser.add_argument('--loop', action='store_true', help='Loop chain sampling instead of parallelizing it (default = False)')
    parser.add_argument('--viter', type=int, default=50_000, help='Number of ADVI steps (default = 50_000)')
    parser.add_argument('--lr', type=float, default=5e-3, help='Adam learning rate for ADVI (default = 5e-3)')
    return parser.parse_args()


class Fitter:
    def __init__(
        self,
        cfg: argparse.Namespace,
        outdir: Path = Path('..', 'outputs', 'data'),
    ) -> None:
        assert cfg.method in ['nuts', 'advi'], 'fit method must be in [nuts, advi]'
        self.cfg = cfg
        self.outdir = outdir

        # determine path to data
        self.cfg.partition = 'test'
        filename = datasetFilename(self.cfg)
        path = Path(self.outdir, filename)
        assert path.exists(), f'{path} does not exist'

        # load batch
        with np.load(path, allow_pickle=True) as batch:
            self.batch = dict(batch)

    def __len__(self):
        return len(self.batch['X'])

    def _get(self, idx: int) -> dict[str, np.ndarray]:
        ''' extract a single dataset from batch and remove padding '''
        assert 0 <= idx < len(self), 'idx out of bounds'
        ds = {k: v[idx] for k,v in self.batch.items()}

        # --- unpad
        d, q, m, n = ds['d'], ds['q'], ds['m'], ds['n']

        # observations
        ds['y'] = ds['y'][:n]
        ds['X'] = ds['X'][:n, :d]
        ds['groups'] = ds['groups'][:n]
        ds['ns'] = ds['ns'][:m]

        # hyperparams
        ds['nu_ffx'] = ds['nu_ffx'][:d]
        ds['tau_ffx'] = ds['tau_ffx'][:d]
        ds['tau_rfx'] = ds['tau_rfx'][:q]

        # params
        ds['ffx'] = ds['ffx'][:d]
        ds['sigma_rfx'] = ds['sigma_rfx'][:q]
        ds['rfx'] = ds['rfx'][:m, :q]
        return ds

    def _pandify(self, ds: dict[str, np.ndarray]) -> pd.DataFrame:
        ''' get observations as dataframe '''
        n, d = ds['n'], ds['d']
        df = pd.DataFrame(index=range(n))
        df['i'] = ds['groups']
        df['y'] = ds['y'][:, None]
        for j in range(1, d):
            df[f'x{j}'] = ds['X'][..., j]
        return df

    def _formulate(self, ds: dict[str, np.ndarray]) -> str:
        ''' setup bambi model formula based on ds '''
        d, q = ds['d'], ds['q']
        fixed = ' + '.join(f'x{j}' for j in range(1, d))
        random = ' + '.join(f'x{j}' for j in range(1, q))
        if not random: # no random slopes
            random = '1'
        out = f'y ~ 1 + {fixed} + ({random} | i)'
        return out

    def _priorize(self, ds: dict[str, np.ndarray]) -> dict[str, bmb.Prior]:
        ''' setup bambi priors based on true priors '''
        d, q = ds['d'], ds['q']
        nu_ffx = ds['nu_ffx']
        tau_ffx = ds['tau_ffx']
        tau_rfx = ds['tau_rfx']
        tau_eps = ds['tau_eps']
        priors = {}

        # fixed effects
        if not cfg.respecify_ffx: # otherwise bambi will infer them from data
            for j in range(d):
                key = 'Intercept' if j == 0 else f'x{j}'
                priors[key] = bmb.Prior('Normal', mu=nu_ffx[j], sigma=tau_ffx[j])

        # random effects variance
        for j in range(q):
            key = '1|i' if j == 0 else f'x{j}|i'
            sigma = bmb.Prior('HalfNormal', sigma=tau_rfx[j])
            priors[key] = bmb.Prior('Normal', mu=0, sigma=sigma)

        # noise variance
        priors['sigma'] = bmb.Prior('HalfStudentT', nu=4, sigma=tau_eps)
        return priors

    def bambify(self, ds: dict[str, np.ndarray]) -> bmb.Model:
        ''' setup bambi model from dataset dict
            optionally allow bambi to setup its own priors for the fixed effects'''
        df = self._pandify(ds)
        form = self._formulate(ds)
        priors = None
        if 'nu_ffx' in ds:
            priors = self._priorize(ds)

        model = bmb.Model(formula=form, data=df, categorical='i', priors=priors)
        model.build()
        return model

    def _extract(self, trace: az.InferenceData, name: str) -> np.ndarray:
        ''' extract {name} samples from posterior '''
        x = trace.posterior[name].to_numpy() # type: ignore
        shape = (x.shape[0] * x.shape[1], ) + x.shape[2:]
        x = x.reshape(shape)
        return x[None, ...]

    def _extractAll(self, trace: az.InferenceData,
                    d: int, q: int, prefix: str) -> dict[str, np.ndarray]:
        ''' extract all posterior samples '''
        # fixed effects
        ffx = []
        for j in range(d):
            key = 'Intercept' if j == 0 else f'x{j}'
            val = self._extract(trace, key)
            ffx.append(val)
        ffx = np.concatenate(ffx, axis=0)

        # variances
        sigma_eps = self._extract(trace, 'sigma')
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
        return {
            f'{prefix}_ffx': ffx,
            f'{prefix}_sigma_eps': sigma_eps,
            f'{prefix}_sigma_rfx': sigma_rfx,
            f'{prefix}_rfx': rfx,
        }

    def _fitNuts(
            self, cfg: argparse.Namespace, ds: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        ''' fit NUTS-based MCMC model and return samples and diagnostics '''
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
        out['nuts_ess'] = summary['ess_bulk'].to_numpy()
        out['nuts_rhat'] = summary['r_hat'].to_numpy()
        out['nuts_divergences'] = trace.sample_stats['diverging'].values.sum(-1)
        out['nuts_duration'] = np.array(t1 - t0)
        return out

    def _fitAdvi(
            self, cfg: argparse.Namespace, ds: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        ''' fit ADVI model (using ADAM) and return samples and diagnostics'''
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
        out['advi_ess'] = summary['ess_bulk'].to_numpy()
        out['advi_duration'] = np.array(t1 - t0)
        return out

    def go(self, idx: int) -> dict[str, np.ndarray]:
        ''' wrapper for fitting a single dataset with index {idx} '''
        ds = self._get(idx)
        if self.cfg.method == 'advi':
            return self._fitAdvi(self.cfg, ds)
        return self._fitNuts(self.cfg, ds)




# -----------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = setup()
    fitter = Fitter(cfg)
    results = fitter.go(0)

