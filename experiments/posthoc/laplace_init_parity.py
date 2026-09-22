"""Local CPU experiment: does the Laplace Newton init source change the posterior?

On a trained GLMM model, compares newton_init in {flow, analytical, cold} for the Laplace
mode search + Rao-Blackwellised redraw:

  - guard-pinned fraction (ll == -1e10): samples whose Newton solve didn't resolve within
    budget -> effectively rejected. Lower is better.
  - mode agreement vs the flow baseline (init-independence check on the converged mode).
  - marginal-likelihood (== IS weight up to shared globals) agreement vs flow.
  - end-to-end sampler check: IS effective-sample-size and IMH acceptance under each init.

Run from repo root:
    uv run python experiments/posthoc/laplace_init_parity.py --family bernoulli
"""

import argparse
import copy
import time
from pathlib import Path

import torch

from metabeta.utils.posterior_eval import loadModel
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler, logMarginalLikelihoodLaplace
from metabeta.posthoc.metropolis import MetropolisSampler

DEV = torch.device('cpu')
CKPTS = {
    'bernoulli': Path('metabeta/outputs/checkpoints/data=large-b-mixed_model=large_seed=4'),
    'poisson': Path('metabeta/outputs/checkpoints/data=large-p-mixed_model=large_seed=6'),
}
DATAS = {
    'bernoulli': Path('metabeta/outputs/data/large-b-sampled/valid.npz'),
    'poisson': Path('metabeta/outputs/data/large-p-sampled/valid.npz'),
}
INITS = ('flow', 'analytical', 'cold')


def run(family: str, n_datasets: int, s: int, seed: int):
    torch.manual_seed(seed)
    model, cfg = loadModel(CKPTS[family], 'best', DEV)
    model.eval()
    lf = cfg.likelihood_family
    print(f'model lf={lf} | d_ffx={model.d_ffx} d_rfx={model.d_rfx} d_corr={model.d_corr}')

    dl = Dataloader(DATAS[family], batch_size=n_datasets)
    batch = toDevice(next(iter(dl)), DEV)
    # materialize the cheap analytical stats exactly as the model does internally, so
    # data['stats']['blup_est'] is present for newton_init='analytical'
    stats = model._dataStatistics(batch)
    assert 'blup_est' in stats, f'no blup_est in stats keys={list(stats)}'
    batch['stats'] = stats
    b, m = batch['X'].shape[0], batch['X'].shape[1]
    print(
        f'batch b={b} m={m} n(max)={batch["X"].shape[2]} | blup_est {tuple(stats["blup_est"].shape)}'
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        proposal = model.estimate(batch, n_samples=s, stats=stats)
    print(f'\nflow proposal draw [s={s}]: {(time.perf_counter()-t0)*1e3:.1f} ms\n')

    # shared, init-independent pieces (built from the same proposal)
    ref = LaplaceImportanceSampler(batch, likelihood_family=lf, newton_init='flow')
    lp, ffx, sigma_eps = ref._logPriorGlobals(proposal)
    common = dict(
        ffx=ffx,
        sigma_rfx=proposal.sigma_rfx,
        sigma_eps=sigma_eps,
        y=ref.y,
        X=ref.X,
        Z=ref.Z,
        mask_n=ref.mask_n,
        mask_m=ref.mask_m,
        likelihood_family=lf,
        L_corr=ref._getLCorr(proposal),
        n_newton=5,
    )

    out = {}
    for init in INITS:
        sampler = LaplaceImportanceSampler(batch, likelihood_family=lf, newton_init=init)
        t0 = time.perf_counter()
        ll, modes, _ = logMarginalLikelihoodLaplace(init=sampler._newtonInit(proposal), **common)
        t = time.perf_counter() - t0
        pinned = ll <= -1e9
        out[init] = dict(ll=ll, modes=modes, pinned=pinned)
        print(
            f'  newton_init={init:11s} mode search {t*1e3:7.1f} ms'
            f' | guard-pinned {pinned.float().mean().item()*100:5.2f}%'
        )

    print('\n  agreement vs newton_init=flow (jointly-resolved samples):')
    fm, fll, fp = out['flow']['modes'], out['flow']['ll'], out['flow']['pinned']
    for init in ('analytical', 'cold'):
        ok = ~(fp | out[init]['pinned'])
        okm = ok[:, None, :, None].expand_as(fm)
        rel = (out[init]['modes'] - fm)[okm].norm() / fm[okm].norm().clamp(min=1e-8)
        dll = (out[init]['ll'] - fll)[ok].abs()
        print(
            f'    {init:11s} rel L2(mode) {rel.item():.2e}'
            f' | |Δ log-weight| mean {dll.mean().item():.3e} max {dll.max().item():.3e}'
        )

    print('\n  end-to-end sampler check (IS ESS%, IMH accept%):')
    n_chains, n_steps = 4, s // 4
    for init in INITS:
        isamp = LaplaceImportanceSampler(batch, likelihood_family=lf, newton_init=init)
        p_is = isamp(copy.deepcopy(proposal))
        w = p_is.is_results.get('weights')
        ess = (
            (w.sum(-1) ** 2 / (w**2).sum(-1)).mean().item() / w.shape[-1] * 100
            if w is not None
            else float('nan')
        )
        imh = MetropolisSampler(
            batch,
            n_chains=n_chains,
            n_steps=n_steps,
            burnin=n_steps // 5,
            mode='laplace',
            likelihood_family=lf,
            newton_init=init,
            n_eff_target=None,
        )
        _, diag = imh(copy.deepcopy(proposal))
        print(
            f'    {init:11s} IS ESS={ess:5.1f}%   IMH accept={diag["accept_rate"].mean().item()*100:5.1f}%'
        )


# fmt: off
def _setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Local parity: Laplace Newton init source on a GLMM model.')
    p.add_argument('--family',      choices=['bernoulli', 'poisson'], default='bernoulli')
    p.add_argument('--n_datasets',  type=int, default=24)
    p.add_argument('--s',           type=int, default=512)
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()
# fmt: on


if __name__ == '__main__':
    run(**vars(_setup()))
