"""Diagnostic: cost profile of the Laplace weight pass on a cached MB pool.

Runs LaplaceImportanceSampler.unnormalizedPosterior (the imhLaplace weight pass — the
same Newton machinery is run a second time for the conditional redraw) on the exact
posterior-sample cache an oracle/real run used, under several Newton budgets, and
reports wall time plus the number of Newton iterations, objective (backtracking)
evaluations, and pinned samples. Use it to separate "the node is slow" from "the
adaptive budget is burning iterations on this pool".

Run from repo root, e.g. against the medium-p oracle pool:
    uv run python experiments/posthoc/laplace_budget_timing.py \
        --pool metabeta/outputs/data/medium-p-sampled/test-all.mb.data=medium-p-mixed_model=large_seed=11_latest_s1000_seed0_k0.npz \
        --data-dir metabeta/outputs/data/medium-p-sampled --likelihood-family 2 --n-datasets 64
"""

import argparse
import inspect
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation import buildBatches, loadData, splitMergedProposal  # noqa: E402

import metabeta.posthoc.laplace_glmm as lg  # noqa: E402
from metabeta.posthoc.laplace_glmm import LaplaceImportanceSampler  # noqa: E402
from metabeta.utils.posterior_cache import loadProposalCache  # noqa: E402

# (label, laplaceRfxModes kwargs); settings whose kwargs the installed code lacks are skipped
SETTINGS = [
    ('default budget', {}),
    ('no extra iterations', {'n_newton_extra': 0}),
    ('no extra, no backtracking (pre-fix)', {'n_newton_extra': 0, 'n_backtrack': 0}),
]


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pool', type=Path, required=True, help='posterior-sample cache (test-*.mb.*.npz)')
    p.add_argument('--data-dir', type=Path, required=True, help='dataset dir holding test.fit.npz')
    p.add_argument('--likelihood-family', type=int, required=True, choices=[1, 2])
    p.add_argument('--n-datasets', type=int, default=32)
    p.add_argument('--n-samples', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4)
    return p.parse_args()
# fmt: on


class Counter:
    """Counts calls to the Newton-step and objective helpers of laplace_glmm."""

    def __init__(self) -> None:
        self.newton = 0
        self.objective = 0

    def __enter__(self) -> 'Counter':
        self._orig = (lg._meanWeightScore, lg._llPerGroup)

        def mws(*a, **k):
            self.newton += 1
            return self._orig[0](*a, **k)

        def llg(*a, **k):
            self.objective += 1
            return self._orig[1](*a, **k)

        lg._meanWeightScore, lg._llPerGroup = mws, llg
        return self

    def __exit__(self, *exc) -> None:
        lg._meanWeightScore, lg._llPerGroup = self._orig


def main() -> None:
    args = setup()
    items, *_ = loadData(args.data_dir / 'test.fit.npz', args.n_datasets)
    batches = buildBatches(items, args.batch_size)
    merged, _ = loadProposalCache(args.pool)
    proposals = splitMergedProposal(merged, batches, args.n_samples)
    n_ds = len(items)
    print(
        f'{args.pool.name}\n{n_ds} datasets x {args.n_samples} samples, torch threads={torch.get_num_threads()}\n'
    )

    accepted = set(inspect.signature(lg.laplaceRfxModes).parameters)
    orig_modes = lg.laplaceRfxModes
    for label, kw in SETTINGS:
        if not set(kw) <= accepted:
            print(f'{label:40s} (not available in this code version)')
            continue

        def patched(*a, **k):
            k.update(kw)
            return orig_modes(*a, **k)

        lg.laplaceRfxModes = patched
        pinned = 0
        with Counter() as c, torch.no_grad():
            t0 = time.perf_counter()
            for p, batch in zip(proposals, batches):
                sampler = LaplaceImportanceSampler(batch, likelihood_family=args.likelihood_family)
                ll, _ = sampler.unnormalizedPosterior(p)
                pinned += int((ll < -1e9).sum())
            dt = time.perf_counter() - t0
        lg.laplaceRfxModes = orig_modes
        n_pass = len(batches)
        print(
            f'{label:40s} {dt:7.1f}s  {dt / n_ds:5.2f}s/ds   '
            f'newton iters/pass={c.newton / n_pass:5.1f}   objective evals/pass={c.objective / n_pass:5.1f}   '
            f'pinned={pinned}/{n_ds * args.n_samples}'
        )


if __name__ == '__main__':
    main()
