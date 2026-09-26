"""Wall-clock latency of the GLMM IMH refinements per dataset: imhLaplace vs imhPM (pseudo-marginal,
IS² weights) vs imhPMr (imhPM + iterated-SIR rfx refresh), on one device.

Per (family, size): one flow pool per dataset (local flow skipped, as runIMH does), then the
three refinements interleaved on the same pool; the first dataset is an untimed warm-up and
every timed region is synchronised on the device. Run on an otherwise idle machine.

Run from the repo root:
    uv run python experiments/posthoc/imh_timing.py --device cuda
    uv run python experiments/posthoc/imh_timing.py --device cpu --sizes small huge
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from metabeta.utils.experiments import DATA_DIR, REPO_ROOT, experimentResultsPath

sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from build_ckpt import BEST_SEEDS, _ckpt_dir  # noqa: E402

from metabeta.models.approximator import Approximator  # noqa: E402
from metabeta.posthoc.metropolis import MetropolisSampler  # noqa: E402
from metabeta.utils.config import ApproximatorConfig  # noqa: E402
from metabeta.utils.dataloader import Collection, collateGrouped, toDevice  # noqa: E402

FAMILIES = {'bernoulli': 1, 'poisson': 2}
MODES = {
    'imhLaplace': dict(n_inner=0),
    'imhPM': dict(n_inner=8),
    'imhPMr': dict(n_inner=8, rfx_refresh=True),
}
N_CHAINS = 4


# fmt: off
def setup() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='cuda')
    p.add_argument('--families', nargs='+', default=list(FAMILIES), choices=list(FAMILIES))
    p.add_argument('--sizes', nargs='+', default=['small', 'medium', 'large', 'huge'])
    p.add_argument('--prefix', default='latest', help='checkpoint prefix (latest = the checkpoint of the paper tables)')
    p.add_argument('--n-datasets', type=int, default=16, help='timed datasets per (family, size)')
    p.add_argument('--n-samples', type=int, default=4000, help='IMH pool size')
    p.add_argument('--out-dir', type=Path, default=experimentResultsPath('evidence'))
    return p.parse_args()
# fmt: on


class Timing:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        torch.set_grad_enabled(False)

    def sync(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()

    def model(self, family: str, size: str) -> Approximator:
        ckpt = _ckpt_dir(family, size, BEST_SEEDS[(family, size)]) / f'{self.args.prefix}.pt'
        payload = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = Approximator(ApproximatorConfig(**payload['model_cfg']))
        model.load_state_dict(payload['model_state'])
        return model.eval().to(self.device)

    def run(self, family: str, size: str) -> list[dict]:
        lf = FAMILIES[family]
        model = self.model(family, size)
        col = Collection(DATA_DIR / f'{size}-{family[0]}-sampled' / 'test.npz', permute=False)
        n_steps = self.args.n_samples // N_CHAINS
        rows = []
        for i in range(self.args.n_datasets + 1):  # dataset 0: warm-up
            batch = toDevice(collateGrouped([col[i]]), self.device)
            pool = model.estimate(batch, n_samples=N_CHAINS * n_steps, local=False)
            for mode, kwargs in MODES.items():
                sampler = MetropolisSampler(
                    batch,
                    n_chains=N_CHAINS,
                    n_steps=n_steps,
                    mode='laplace',
                    likelihood_family=lf,
                    n_eff_target=None,
                    **kwargs,
                )
                self.sync()
                t0 = time.perf_counter()
                _, diag = sampler(pool.slice_b(0, 1))
                self.sync()
                if i:
                    rows.append(
                        {
                            'family': family,
                            'size': size,
                            'idx': i,
                            'm': int(batch['m']),
                            'q': int(batch['mask_q'].sum()),
                            'mode': mode,
                            'sec': time.perf_counter() - t0,
                            'accept': float(diag['accept_rate'].mean()),
                        }
                    )
        return rows

    def go(self) -> pd.DataFrame:
        rows = []
        for family in self.args.families:
            for size in self.args.sizes:
                rows += self.run(family, size)
                df = pd.DataFrame(rows)
                last = df[(df['family'] == family) & (df['size'] == size)]
                med = last.groupby('mode')['sec'].median()
                print(
                    f'{family:9s} {size:6s} '
                    + '  '.join(f'{m} {med[m]:.3f}s' for m in MODES)
                    + f'   PM/Lap {med["imhPM"] / med["imhLaplace"]:.2f}'
                    + f'  PMr/Lap {med["imhPMr"] / med["imhLaplace"]:.2f}',
                    flush=True,
                )
        return pd.DataFrame(rows)

    def save(self, df: pd.DataFrame) -> None:
        self.args.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f'imh_timing_{self.device.type}_s{self.args.n_samples}'
        df.to_csv(self.args.out_dir / f'{stem}.csv', index=False)
        table = df.pivot_table(
            index=['family', 'size'], columns='mode', values='sec', aggfunc='median'
        )
        (self.args.out_dir / f'{stem}.md').write_text(
            f'# IMH latency per dataset ({self.device}, pool {self.args.n_samples}), median seconds\n\n'
            + table[list(MODES)].to_markdown(floatfmt='.3f')
            + '\n'
        )
        print(f'[saved] {self.args.out_dir / stem}.{{csv,md}}')


# =============================================================================
if __name__ == '__main__':
    args = setup()
    timing = Timing(args)
    timing.save(timing.go())
