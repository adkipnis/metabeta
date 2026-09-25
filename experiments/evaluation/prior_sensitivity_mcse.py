"""
E2 follow-up: how much of MB's key-effect disagreement with NUTS is Monte Carlo noise?

Step 1 (--stages mcse): for every NUTS-converged Normal-slope prior of the slope variant, the ESS
and the MCSE of the key-effect median on both sides (arviz ess / mcse(method='median')), from the
4 IMH chains x 1000 draws of the cached MB run and the 4 NUTS chains x 1000 draws.  Both store
their draws chain-major: MetropolisSampler._runChains reshapes (b, C, T_post) -> (b, C*T_post),
utils/pymc.extractSingle flattens (chains, draws).  z = |med_MB - med_NUTS| / sqrt(mcse_MB^2 +
mcse_NUTS^2).

Step 2 (--stages rerun --seed S): MB (flow + IMH) again on the Mac CPU for the Normal-slope blocks,
the same batched Api.sample calls as prior_sensitivity.py's mb stage (one call per (ffx family, SD
family) block, setSeed before each call, so dropping the Student-t blocks changes nothing); keeps
the key-effect draws in mcse_cpu_seed{S}_key.pt.  --stages spread compares the GPU run (seed 0) with
the reruns.

Mac only, eval-time only: reads mb_cuda.pt, the NUTS fits and the prior_sensitivity CSVs.
    uv run python experiments/evaluation/prior_sensitivity_mcse.py --stages mcse
    uv run python experiments/evaluation/prior_sensitivity_mcse.py --stages rerun --seed 1
    uv run python experiments/evaluation/prior_sensitivity_mcse.py --stages spread
"""

import argparse
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import torch

from metabeta.utils.api import parseFormula
from metabeta.utils.experiments import DATA_DIR, RESULTS_DIR
from metabeta.utils.sampling import setSeed

# sibling experiment script (this directory is sys.path[0] at run time)
from prior_sensitivity import (
    DATASETS,
    FIRE_ACCEPT,
    FIRE_KHAT,
    KEY_Z_MAX,
    WIDTH_RATIO_RANGE,
    PriorSensitivity,
)

# ==============================================================================
# Globals
# ==============================================================================

RFX = 'slope'
NAMES = tuple(DATASETS)
CHAINS = 4


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', nargs='+', required=True, choices=['mcse', 'rerun', 'spread'])
    parser.add_argument('--seed', type=int, default=1, help='rerun: seed of the CPU MB run')
    parser.add_argument('--reruns', type=int, nargs='+', default=[1, 2], help='spread: CPU seeds to compare')
    return parser.parse_args()
# fmt: on


def dataDir(name: str) -> Path:
    return DATA_DIR / f'e2-{name}-{RFX}'


def chainStats(x: np.ndarray) -> tuple[float, float, float]:
    """(median, ESS, MCSE of the median) of chain-major draws (C*T,)."""
    ch = x.reshape(CHAINS, -1)
    return (
        float(np.median(x)),
        float(az.ess(ch, method='bulk')),
        float(az.mcse(ch, method='median')),
    )


# ==============================================================================
# Pipeline
# ==============================================================================


class McseCheck:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg

    def keyIndex(self, name: str) -> int:
        terms = list(parseFormula(DATASETS[name]['formula'][RFX]).fixed_terms)
        return 1 + terms.index(DATASETS[name]['key'])

    def mcse(self) -> pd.DataFrame:
        rows = []
        for name in NAMES:
            ki = self.keyIndex(name)
            df = pd.read_csv(RESULTS_DIR / f'prior_sensitivity_{RFX}_{name}.csv')
            df = df[df.ffx_family == 'normal']
            nuts = df[df.method == 'NUTS'].set_index('point')
            mbrow = df[df.method == 'MB'].set_index('point')
            conv = nuts.index[nuts.nuts_converged.astype(bool)]
            mb = torch.load(dataDir(name) / 'mb_cuda.pt', weights_only=False)
            for block in mb['blocks']:
                g = block['MB']['data']['global']['samples']  # (B, 4000, D_g)
                for j, point in enumerate(block['points']):
                    if point not in conv:
                        continue
                    x_mb = g[j, :, ki].numpy()
                    with np.load(dataDir(name) / 'fits' / f'test_nuts_{point:03d}.npz') as f:
                        x_nuts = f['nuts_ffx'][ki]
                    med_m, ess_m, mcse_m = chainStats(x_mb)
                    med_n, ess_n, mcse_n = chainStats(x_nuts)
                    r = mbrow.loc[point]
                    sd_n = x_nuts.std()
                    rows.append({
                        'dataset': name, 'point': int(point), 'tau_beta': r.tau_beta,
                        'sd_prior': f'{r.sigma_family}({r.tau_sigma:g})',
                        'flagged': bool(r.flagged), 'accept': r.accept_rate, 'k_hat': r.k_hat,
                        'fired': bool(r.accept_rate < FIRE_ACCEPT or r.k_hat > FIRE_KHAT),
                        'shift_sd': abs(med_m - med_n) / sd_n, 'shift_sd_csv': r.key_z,
                        'width_ratio': r.key_width_ratio,
                        'ess_mb': ess_m, 'ess_nuts': ess_n,
                        'mcse_mb_sd': mcse_m / sd_n, 'mcse_nuts_sd': mcse_n / sd_n,
                        'z': abs(med_m - med_n) / np.hypot(mcse_m, mcse_n),
                    })  # fmt: skip
        out = pd.DataFrame(rows)
        if not np.allclose(out.shift_sd, out.shift_sd_csv, atol=1e-4):
            raise ValueError('key-effect shift does not reproduce the analysis CSV')
        out.to_csv(RESULTS_DIR / 'prior_sensitivity_mcse.csv', index=False)
        return out

    def rerun(self) -> None:
        from metabeta.models.api import Api
        from metabeta.utils.constants import LIKELIHOOD_FAMILIES

        ps = PriorSensitivity(argparse.Namespace(datasets=list(NAMES), rfx=RFX))
        for name in NAMES:
            ki, meta = self.keyIndex(name), ps.meta[name]
            api = Api.from_pretrained(
                LIKELIHOOD_FAMILIES[meta['lf']], device='cpu', batch_size=None
            )
            blocks = []
            for block in ps.blocks(name):
                if block.ffx_family.iloc[0] != 'normal':
                    continue
                priors = {f'p{i:03d}': ps.prior(name, point) for i, point in block.iterrows()}
                setSeed(self.cfg.seed)
                t0 = time.perf_counter()
                res = api.sample(
                    ps.data[name],
                    formula=ps.formula(name),
                    priors=priors,
                    n_samples=4000,
                    refine=True,
                )
                g = res.proposal.samples_g  # (B, 4000, D_g)
                blocks.append({
                    'points': block.index.to_numpy(), 'key': g[:, :, ki].cpu().clone(),
                    'accept_rate': res.safeguards['accept_rate'], 'wall_s': time.perf_counter() - t0,
                })  # fmt: skip
                print(
                    f'{name} {block.sigma_family.iloc[0]} seed {self.cfg.seed}: {blocks[-1]["wall_s"]:.1f} s',
                    flush=True,
                )
            torch.save(blocks, dataDir(name) / f'mcse_cpu_seed{self.cfg.seed}_key.pt')

    def spread(self) -> pd.DataFrame:
        """Key-effect median, E2 flag and IMH acceptance of the GPU run (seed 0) and the CPU reruns."""
        base = pd.read_csv(RESULTS_DIR / 'prior_sensitivity_mcse.csv')
        rows = []
        for name in NAMES:
            ki = self.keyIndex(name)
            mb = torch.load(dataDir(name) / 'mb_cuda.pt', weights_only=False)
            runs = {0: {}}  # seed -> point -> (key draws, acceptance)
            for block in mb['blocks']:
                for j, p in enumerate(block['points']):
                    runs[0][int(p)] = (
                        block['MB']['data']['global']['samples'][j, :, ki].numpy(),
                        block['MB']['accept_rate'][j],
                    )
            for seed in self.cfg.reruns:
                runs[seed] = {}
                for block in torch.load(
                    dataDir(name) / f'mcse_cpu_seed{seed}_key.pt', weights_only=False
                ):
                    for j, p in enumerate(block['points']):
                        runs[seed][int(p)] = (block['key'][j].numpy(), block['accept_rate'][j])
            for point in base.point[base.dataset == name]:
                with np.load(dataDir(name) / 'fits' / f'test_nuts_{point:03d}.npz') as f:
                    x_n = f['nuts_ffx'][ki]
                width = lambda x: np.quantile(x, 0.95) - np.quantile(x, 0.05)
                row = {'dataset': name, 'point': int(point)}
                meds = []
                for seed, run in runs.items():
                    x, acc = run[int(point)]
                    z, wr = abs(np.median(x) - np.median(x_n)) / x_n.std(), width(x) / width(x_n)
                    meds.append(np.median(x))
                    row |= {f'shift_s{seed}': z, f'width_s{seed}': wr, f'accept_s{seed}': float(acc),
                            f'flag_s{seed}': bool(z > KEY_Z_MAX or not WIDTH_RATIO_RANGE[0] <= wr <= WIDTH_RATIO_RANGE[1])}  # fmt: skip
                row['spread_sd'] = (max(meds) - min(meds)) / x_n.std()
                rows.append(row)
        out = base.merge(pd.DataFrame(rows), on=['dataset', 'point'])
        if not (out.flag_s0 == out.flagged).all():
            raise ValueError('GPU-run flags do not reproduce the analysis CSV')
        out.to_csv(RESULTS_DIR / 'prior_sensitivity_mcse.csv', index=False)
        return out


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    cfg = setup()
    check = McseCheck(cfg)
    if 'mcse' in cfg.stages:
        check.mcse()
    if 'rerun' in cfg.stages:
        check.rerun()
    if 'spread' in cfg.stages:
        check.spread()
