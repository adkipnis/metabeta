"""Which simulated Poisson test datasets did the simulator clip?

The Poisson simulator draws y from exp(min(η, POISSON_ETA_CLIP_MAX)), and MB, IS² and AGQ
evaluate that clipped likelihood, but PyMC's NUTS, ADVI and Pathfinder fit the exact
pm.Poisson(exp(η)). On a dataset where any true η exceeds the clip the NUTS reference
therefore targets another posterior. evidence.py drops those rows (etaClipped); this script
applies the same criterion to every simulated Poisson test set under DATA_DIR (oracle and
stress conditions; real-outcome sets are skipped) and records the reference fit's
diagnostics next to each flag.

Output: experiments/results/eta_clipped/eta_clipped_p.csv, one row per dataset with
  dir, idx, max_eta, frac_clipped, eta_clipped, nuts2_converged, nuts2_max_rhat,
  nuts2_min_ess, nuts2_div (total divergent transitions), nuts2_level;
and eta_clipped_p.md, the per-directory counts.

Run from the repo root:
    uv run python experiments/posthoc/eta_clipped.py
"""

import numpy as np
import pandas as pd

# evidence.py sits in this directory (on sys.path[0] at run time).
from evidence import etaClipped

from metabeta.utils.dataloader import Collection
from metabeta.utils.evaluation import NUTS_DIAG_KEYS, nutsConverged
from metabeta.utils.experiments import DATA_DIR, experimentResultsPath
from metabeta.utils.families import POISSON_ETA_CLIP_MAX
from metabeta.utils.fits import REFERENCE_TAG, fitPath, loadFits

OUT_DIR = experimentResultsPath('eta_clipped')


def flagFile(name: str) -> pd.DataFrame:
    path = DATA_DIR / name / 'test.npz'
    col = Collection(path, permute=False)
    rows = []
    for i in range(len(col)):
        item = col[i]
        eta = item['X'] @ item['ffx'] + (item['Z'] * item['rfx'][item['groups'].astype(int)]).sum(-1)
        rows.append(
            {
                'dir': name,
                'idx': i,
                'max_eta': float(eta.max()),
                'frac_clipped': float(np.mean(eta > POISSON_ETA_CLIP_MAX)),
                'eta_clipped': etaClipped(item),
            }
        )
    df = pd.DataFrame(rows)
    if fitPath(path, REFERENCE_TAG).exists():
        ref = REFERENCE_TAG
        diag = loadFits(path, ref, keys=[f'{ref}_{k}' for k in NUTS_DIAG_KEYS])
        rhat = np.where(diag[f'{ref}_rhat'] > 0, diag[f'{ref}_rhat'], np.nan)
        ess = np.where(diag[f'{ref}_ess'] > 0, diag[f'{ref}_ess'], np.nan)
        df['nuts2_converged'] = nutsConverged(diag, ref)
        df['nuts2_max_rhat'] = np.nanmax(rhat, axis=-1)
        df['nuts2_min_ess'] = np.nanmin(ess, axis=-1)
        df['nuts2_div'] = np.asarray(diag[f'{ref}_divergences']).sum(-1)
        df['nuts2_level'] = np.asarray(diag[f'{ref}_level']).reshape(-1)
    return df


def main() -> None:
    names = sorted(
        p.name
        for p in DATA_DIR.iterdir()
        if '-p-' in p.name and not p.name.endswith('-real') and (p / 'test.npz').exists()
    )
    df = pd.concat([flagFile(name) for name in names], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'eta_clipped_p.csv', index=False)

    clipped = df[df['eta_clipped']]
    counts = df.groupby('dir', sort=False).agg(
        n=('idx', 'size'),
        clipped=('eta_clipped', 'sum'),
        conv=('nuts2_converged', 'sum'),
    )
    counts['clipped_conv'] = clipped.groupby('dir')['nuts2_converged'].sum()
    counts = counts.fillna(0).astype(int)
    lines = ['# Simulator-clipped Poisson test datasets (max true η > %g)' % POISSON_ETA_CLIP_MAX]
    lines += ['', counts.to_markdown(), '', '## Clipped datasets', '']
    lines += [clipped.drop(columns='eta_clipped').to_markdown(index=False, floatfmt='.4g')]
    (OUT_DIR / 'eta_clipped_p.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
