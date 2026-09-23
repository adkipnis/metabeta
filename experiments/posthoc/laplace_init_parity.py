"""Local CPU experiment: does the start of the Laplace Newton mode search change the posterior?

On a trained GLMM model, compares three starts of the per-group mode search behind the
Laplace weights and the Rao-Blackwellised rfx redraw, all on the same global draws:

  flow        the flow's rfx draws                 model.estimate(...)
  analytical  the analytical rfx point estimate    model.estimate(..., local=False)
  cold        zeros

  - guard-pinned fraction (ll == -1e10): samples whose Newton solve didn't resolve within
    budget -> effectively rejected. Lower is better.
  - mode agreement vs the flow baseline (init-independence check on the converged mode).
  - marginal-likelihood (== IS weight up to shared globals) agreement vs flow.
  - end-to-end sampler check: IS effective-sample-size and IMH acceptance under each start.

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


def zeroLocal(proposal):
    """Copy of ``proposal`` whose rfx are zeros (cold Newton start)."""
    out = copy.deepcopy(proposal)
    out.data['local']['samples'] = torch.zeros_like(out.samples_l)
    return out


@torch.no_grad()
def drawProposals(model, batch, stats, s, seed):
    """{start: Proposal} sharing one set of global draws; only the rfx differ."""
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    flow = model.estimate(batch, n_samples=s, stats=stats)
    print(f'\nflow proposal draw [s={s}]: {(time.perf_counter()-t0)*1e3:.1f} ms')
    torch.manual_seed(seed)  # the globals are drawn first, so re-seeding reproduces them
    t0 = time.perf_counter()
    analytical = model.estimate(batch, n_samples=s, stats=stats, local=False)
    print(f'globals-only draw    [s={s}]: {(time.perf_counter()-t0)*1e3:.1f} ms\n')
    assert torch.equal(flow.samples_g, analytical.samples_g)
    return {'flow': flow, 'analytical': analytical, 'cold': zeroLocal(flow)}


def run(family: str, n_datasets: int, s: int, seed: int):
    model, cfg = loadModel(CKPTS[family], 'best', DEV)
    model.eval()
    lf = cfg.likelihood_family
    print(f'model lf={lf} | d_ffx={model.d_ffx} d_rfx={model.d_rfx} d_corr={model.d_corr}')

    dl = Dataloader(DATAS[family], batch_size=n_datasets)
    batch = toDevice(next(iter(dl)), DEV)
    stats = model._dataStatistics(batch)
    b, m, n = batch['X'].shape[:3]
    print(f'batch b={b} m={m} n(max)={n} | blup_est {tuple(stats["blup_est"].shape)}')
    proposals = drawProposals(model, batch, stats, s, seed)

    # shared, start-independent pieces (built from the shared globals)
    ref = LaplaceImportanceSampler(batch, likelihood_family=lf)
    flow = proposals['flow']
    _, ffx, sigma_eps = ref._logPriorGlobals(flow)
    common = dict(
        ffx=ffx,
        sigma_rfx=flow.sigma_rfx,
        sigma_eps=sigma_eps,
        y=ref.y,
        X=ref.X,
        Z=ref.Z,
        mask_n=ref.mask_n,
        mask_m=ref.mask_m,
        likelihood_family=lf,
        L_corr=ref._getLCorr(flow),
        n_newton=5,
    )

    out = {}
    for name, p in proposals.items():
        t0 = time.perf_counter()
        ll, modes, _ = logMarginalLikelihoodLaplace(init=p.rfx, **common)
        t = time.perf_counter() - t0
        pinned = ll <= -1e9
        out[name] = dict(ll=ll, modes=modes, pinned=pinned)
        print(
            f'  start={name:11s} mode search {t*1e3:7.1f} ms'
            f' | guard-pinned {pinned.float().mean().item()*100:5.2f}%'
        )

    print('\n  agreement vs start=flow (jointly-resolved samples):')
    fm, fll, fp = out['flow']['modes'], out['flow']['ll'], out['flow']['pinned']
    for name in ('analytical', 'cold'):
        ok = ~(fp | out[name]['pinned'])
        okm = ok[:, None, :, None].expand_as(fm)
        rel = (out[name]['modes'] - fm)[okm].norm() / fm[okm].norm().clamp(min=1e-8)
        dll = (out[name]['ll'] - fll)[ok].abs()
        print(
            f'    {name:11s} rel L2(mode) {rel.item():.2e}'
            f' | |Δ log-weight| mean {dll.mean().item():.3e} max {dll.max().item():.3e}'
        )

    print('\n  end-to-end sampler check (IS ESS%, IMH accept%):')
    n_chains, n_steps = 4, s // 4
    for name, p in proposals.items():
        p_is = LaplaceImportanceSampler(batch, likelihood_family=lf)(copy.deepcopy(p))
        w = p_is.is_results['weights']
        ess = (w.sum(-1) ** 2 / (w**2).sum(-1)).mean().item() / w.shape[-1] * 100
        imh = MetropolisSampler(
            batch,
            n_chains=n_chains,
            n_steps=n_steps,
            burnin=n_steps // 5,
            mode='laplace',
            likelihood_family=lf,
            n_eff_target=None,
        )
        _, diag = imh(copy.deepcopy(p))
        print(
            f'    {name:11s} IS ESS={ess:5.1f}%   IMH accept={diag["accept_rate"].mean().item()*100:5.1f}%'
        )


# fmt: off
def _setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Local parity: Laplace Newton start on a GLMM model.')
    p.add_argument('--family',      choices=['bernoulli', 'poisson'], default='bernoulli')
    p.add_argument('--n_datasets',  type=int, default=24)
    p.add_argument('--s',           type=int, default=512)
    p.add_argument('--seed',        type=int, default=0)
    return p.parse_args()
# fmt: on


if __name__ == '__main__':
    run(**vars(_setup()))
