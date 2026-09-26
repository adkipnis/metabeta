"""IS² tuning: noise sd(log p̂(y|θ)) and cost per pass over the inner draw count K and the
defensive-mixture weight α (posthoc/laplace_glmm.py, logMarginalLikelihoodIS2).

Per dataset, a flow pool of S draws is weighted once (PSIS on IS² weights at K=32, α=0.05);
for every (K, α) two independent IS² passes over the pool give sd(log p̂) = sqrt(Σ_s w_s
(a_s − b_s)² / 2), the posterior-averaged noise of the log-likelihood estimate that Tran et
al. (arXiv:1309.3339) tune K by, plus the wall time of one pass.

Run from the repo root:
    uv run python experiments/posthoc/is2_tuning.py --family bernoulli --size small --n-datasets 16
"""

import argparse
import sys
import time
from pathlib import Path

import arviz as az
import pandas as pd
import torch

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT, experimentResultsPath

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

from metabeta.models.approximator import Approximator  # noqa: E402
from metabeta.posthoc.laplace_glmm import (  # noqa: E402
    LaplaceImportanceSampler,
    logMarginalLikelihoodIS2,
)
from metabeta.utils.config import ApproximatorConfig  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped  # noqa: E402

FAMILIES = {'bernoulli': 1, 'poisson': 2}


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--family', default='bernoulli', choices=list(FAMILIES))
    p.add_argument('--size', default='small', choices=['small', 'medium', 'large', 'huge'])
    p.add_argument('--prefix', default='latest', help='checkpoint prefix (latest = the checkpoint of the paper tables)')
    p.add_argument('--n-datasets', type=int, default=16)
    p.add_argument('--n-samples', type=int, default=1000, help='flow pool size S')
    p.add_argument('--inner', nargs='+', type=int, default=[2, 4, 8, 16, 32])
    p.add_argument('--defensive', nargs='+', type=float, default=[0.0, 0.01, 0.05, 0.1])
    p.add_argument('--out-dir', type=Path, default=experimentResultsPath('evidence'))
    return p.parse_args()
# fmt: on


class Tuning:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lf = FAMILIES[args.family]
        seed = BEST_SEEDS[(args.family, args.size)]
        payload = torch.load(
            _ckpt_dir(args.family, args.size, seed) / f'{args.prefix}.pt',
            map_location='cpu',
            weights_only=False,
        )
        self.model = Approximator(ApproximatorConfig(**payload['model_cfg']))
        self.model.load_state_dict(payload['model_state'])
        self.model.eval()
        path = DATA_DIR / f'{args.size}-{args.family[0]}-sampled' / 'test.npz'
        self.col = Collection(path, permute=False)

    def dataset(self, i: int) -> list[dict]:
        batch = collateGrouped([self.col[i]])
        torch.manual_seed(i)
        with torch.no_grad():
            proposal = self.model.estimate(batch, n_samples=self.args.n_samples)
        batch64 = {
            k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in batch.items()
        }
        proposal.data['global']['samples'] = proposal.samples_g.double()
        proposal.data['global']['log_prob'] = proposal.log_prob_g.double()
        proposal.data['local']['samples'] = proposal.samples_l.double()
        sampler = LaplaceImportanceSampler(
            batch64, n_inner=32, corr_prior=True, likelihood_family=self.lf
        )
        ll, lp = sampler.unnormalizedPosterior(proposal)
        log_w, _ = az.psislw((ll + lp - proposal.log_prob_g).numpy())
        w = torch.softmax(torch.as_tensor(log_w[0]), -1)  # (S,)

        _, ffx, sigma_eps = sampler._logPriorGlobals(proposal)
        args = (
            ffx,
            proposal.sigma_rfx,
            sigma_eps,
            sampler.y,
            sampler.X,
            sampler.Z,
            sampler.mask_n,
            sampler.mask_m,
            self.lf,
        )
        kwargs = dict(L_corr=sampler._getLCorr(proposal), init=proposal.rfx)
        rows = []
        for k in self.args.inner:
            for alpha in self.args.defensive:
                t0 = time.perf_counter()
                a, *_ = logMarginalLikelihoodIS2(*args, k, defensive=alpha, **kwargs)
                dt = time.perf_counter() - t0
                b, *_ = logMarginalLikelihoodIS2(*args, k, defensive=alpha, **kwargs)
                sd = float((w * (a - b)[0].square() / 2).sum().sqrt())
                rows.append(
                    {'idx': i, 'm': int(batch['m']), 'K': k, 'alpha': alpha, 'sd': sd, 'sec': dt}
                )
        return rows

    def go(self) -> pd.DataFrame:
        n = min(self.args.n_datasets, len(self.col))
        rows = []
        for i in range(n):
            rows += self.dataset(i)
            print(f'ds={i} done', flush=True)
        return pd.DataFrame(rows)

    def report(self, df: pd.DataFrame) -> None:
        stem = f'is2_tuning_{self.args.family}_{self.args.size}_n{df["idx"].nunique()}'
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.args.out_dir / f'{stem}.csv', index=False)
        sd = df.pivot_table(index='K', columns='alpha', values='sd', aggfunc='median')
        q90 = df.pivot_table(
            index='K', columns='alpha', values='sd', aggfunc=lambda x: x.quantile(0.9)
        )
        sec = df.groupby('K')['sec'].median()
        md = '\n'.join(
            [
                f'# IS² tuning: {self.args.family} ({self.args.size}), S={self.args.n_samples}',
                '',
                '## median sd(log p̂) over datasets (rows K, columns α)',
                '',
                sd.to_markdown(floatfmt='.3f'),
                '',
                '## q90 sd(log p̂)',
                '',
                q90.to_markdown(floatfmt='.3f'),
                '',
                '## median seconds per IS² pass',
                '',
                sec.to_frame().to_markdown(floatfmt='.3f'),
                '',
            ]
        )
        (self.args.out_dir / f'{stem}.md').write_text(md)
        print(md)


# =============================================================================
if __name__ == '__main__':
    args = setup()
    tuning = Tuning(args)
    tuning.report(tuning.go())
