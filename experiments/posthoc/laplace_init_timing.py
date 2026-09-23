"""CPU timing: what skipping the local flow saves at discrete-GLMM MB inference.

Stage-by-stage wall time of the MB pipeline on a trained GLMM model, split by regime
(deep = few groups / many obs, wide = many groups / few obs):

  summarize  : Set-Transformer summaries                          (shared by both configs)
  global (g) : posterior_g.sample                                 (shared)
  local  (A) : posterior_l.sample                                 (the stage MB_skip drops)
  laplace(B) : MetropolisSampler(mode='laplace')                  (mode search + chain + redraw)

  MB_full = summarize + g + A + B     proposal: model.estimate(...)               (flow rfx)
  MB_skip = summarize + g +     B     proposal: model.estimate(..., local=False)  (analytical rfx)

Cache-free by construction: both proposals are drawn live, so — unlike the ablation harness,
which caches the neural-posterior draw and times only the Laplace refinement — the local-flow
stage is actually measured. Accuracy parity is laplace_init_parity.py's job.

Run from repo root:
    uv run python experiments/posthoc/laplace_init_timing.py --family bernoulli --size large
"""

import argparse
import sys
import time

import numpy as np
import torch

from metabeta.utils.experiments import REPO_ROOT, loadApproximator
from metabeta.utils.templates import loadConfigFromCheckpoint
from metabeta.utils.dataloader import Dataloader, toDevice
from metabeta.posthoc.metropolis import MetropolisSampler

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, FAMILY_INITIAL, _ckpt_dir  # noqa: E402

DEV = torch.device('cpu')
DATA_ROOT = REPO_ROOT / 'metabeta' / 'outputs' / 'data'


def loadModel(ckpt_dir, prefix, device):
    """Lean model load (config + weights) — avoids posterior_eval, which pulls the eval/
    plotting stack (arviz/matplotlib). Mirrors posterior_eval.loadModel."""
    cfg = argparse.Namespace(**loadConfigFromCheckpoint(ckpt_dir))
    return loadApproximator(cfg, device, ckpt_dir, prefix), cfg


def resolvePaths(family: str, size: str):
    """(ckpt_dir, valid.npz) for a (family, size), via the ablation's BEST_SEEDS mapping."""
    seed = BEST_SEEDS[(family, size)]
    ckpt = _ckpt_dir(family, size, seed)
    data = DATA_ROOT / f'{size}-{FAMILY_INITIAL[family]}-sampled' / 'valid.npz'
    return ckpt, data


def timeit(fn, reps=3):
    fn()  # warmup
    return min(_time(fn) for _ in range(reps))


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


@torch.no_grad()
def stageTimes(model, sub, stats, s):
    """Per-stage wall time (seconds, whole subset) for summarize / global / local."""
    t_sum = timeit(lambda: model.summarize(sub, stats=stats))
    summary_g, summary_l = model.summarize(sub, stats=stats)
    mask_g = model._masks(sub, local=False)
    t_g = timeit(lambda: model.posterior_g.sample(s, context=summary_g, mask=mask_g))
    samples_g, _ = model.posterior_g.sample(s, context=summary_g, mask=mask_g)
    mask_l = model._masks(sub, local=True)
    b, m, q = mask_l.shape
    mask_l = mask_l.unsqueeze(-2).expand(b, m, s, q)
    context_l = model._localContext(summary_l, samples_g, sub)
    t_l = timeit(lambda: model.posterior_l.sample(1, context=context_l, mask=mask_l))
    return t_sum, t_g, t_l


def imhTime(sub, proposal, lf, s):
    """(seconds, mean acceptance %) of the Laplace IMH refinement of ``proposal``."""
    n_chains, n_steps = 4, s // 4
    sampler = MetropolisSampler(
        sub,
        n_chains=n_chains,
        n_steps=n_steps,
        burnin=n_steps // 5,
        mode='laplace',
        likelihood_family=lf,
        n_eff_target=None,
    )
    t = timeit(lambda: sampler(proposal), reps=2)
    _, diag = sampler(proposal)
    return t, diag['accept_rate'].mean().item() * 100


def run(family: str, size: str, k: int, s: int, seed: int, prefix: str):
    torch.manual_seed(seed)
    ckpt, data_path = resolvePaths(family, size)
    model, cfg = loadModel(ckpt, prefix, DEV)
    model.eval()
    lf = cfg.likelihood_family

    # scan batches (padded per-batch, so memory stays bounded) and keep the batch with the
    # smallest median m (deep) and the largest (wide) — the valid split spans m ~ 13..198
    dl = Dataloader(data_path, batch_size=k)
    deep_b = wide_b = None
    deep_med, wide_med = 1e9, -1.0
    for batch in dl:
        med = batch['mask_m'].squeeze(-1).sum(1).median().item()
        if med < deep_med:
            deep_med, deep_b = med, toDevice(batch, DEV)
        if med > wide_med:
            wide_med, wide_b = med, toDevice(batch, DEV)
    regimes = {'deep (few m)': deep_b, 'wide (many m)': wide_b}

    print(f'\n===== {family} {size} (lf={lf}) | s={s}, {k} datasets/regime =====')
    header = (
        f'{"regime":14s} {"m(med)":>7s} {"n_tot":>7s} | '
        f'{"summ":>7s} {"global":>7s} {"local A":>8s} {"lap B":>7s} | '
        f'{"MB_full":>8s} {"MB_skip":>8s} {"save":>6s}  accept f/a'
    )
    print(header)
    print('-' * len(header))
    for name, sub in regimes.items():
        stats = model._dataStatistics(sub)
        m_sub = sub['mask_m'].squeeze(-1).sum(1).cpu().numpy()
        ntot_sub = sub['mask_n'].sum((1, 2)).cpu().numpy()
        b = sub['X'].shape[0]

        t_sum, t_g, t_l = stageTimes(model, sub, stats, s)
        prop_full = model.estimate(sub, n_samples=s, stats=stats)
        prop_skip = model.estimate(sub, n_samples=s, stats=stats, local=False)
        t_b, acc_f = imhTime(sub, prop_full, lf, s)
        _, acc_a = imhTime(sub, prop_skip, lf, s)

        mb_full = (t_sum + t_g + t_l + t_b) / b * 1e3
        mb_skip = (t_sum + t_g + t_b) / b * 1e3  # B's cost does not depend on its start
        save = (1 - mb_skip / mb_full) * 100
        print(
            f'{name:14s} {np.median(m_sub):7.0f} {np.median(ntot_sub):7.0f} | '
            f'{t_sum/b*1e3:7.1f} {t_g/b*1e3:7.1f} {t_l/b*1e3:8.1f} {t_b/b*1e3:7.1f} | '
            f'{mb_full:8.1f} {mb_skip:8.1f} {save:5.1f}%  {acc_f:4.1f}/{acc_a:4.1f}'
        )


# fmt: off
def _setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Timing: local flow vs Laplace at GLMM inference (cache-free).')
    p.add_argument('--family', choices=['bernoulli', 'poisson'], default='bernoulli')
    p.add_argument('--size',   choices=['small', 'medium', 'large', 'huge'], default='large')
    p.add_argument('--k',      type=int, default=24, help='datasets per regime')
    p.add_argument('--s',      type=int, default=512, help='posterior pool size')
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--prefix', type=str, default='latest', help='checkpoint prefix (best/latest)')
    return p.parse_args()
# fmt: on


if __name__ == '__main__':
    run(**vars(_setup()))
