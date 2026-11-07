"""Export the two MixedBench sources of the real-data group budget (group_budget_real_sets.py).

Runs in MixedBench's environment, since metabeta does not depend on it, and writes each source
in the layout of metabeta/datasets/preprocessed (X without intercept, y, groups sorted 0..m-1,
ns, columns), plus the original group names. The output must not go to preprocessed/test/,
whose files form the held-out pool of the real test sets.

- stroop (afex, MixedBench afex/stroop/2): log(rt) ~ condition * congruency, grouped by
  participant, rows with a recorded rt; the documented fixed effects, its random effects
  reduced to the participant intercept.
- yelp (BeyondMLR, MixedBench beyondmlr/data/yelp.csv/2): I(useful > 0) on the five documented
  covariates, grouped by restaurant name; the crossed user effect is left out.

Usage (from ~/Code/mixedbench):
    uv run python -I <metabeta>/experiments/simulation/group_budget_real_export.py \
        registry/cache <metabeta>/metabeta/outputs/data/group-budget-sources
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mixedbench.registry import Tables, rows


def stroop(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df[df['rt'].notna()]
    deplete = (df['condition'] == 'deplete').astype(float)
    incongruent = (df['congruency'] == 'incongruent').astype(float)
    x = pd.DataFrame(
        {
            'deplete': deplete,
            'incongruent': incongruent,
            'deplete:incongruent': deplete * incongruent,
        }
    )
    return x, np.log(df['rt']), df['pno'].astype(str)


def yelp(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = ['stars_given', 'length', 'year', 'rev_count_user', 'rev_count_biz']
    return df[cols].astype(float), (df['useful'] > 0).astype(int), df['name'].astype(str)


SOURCES = {
    'stroop': ('afex/stroop/2', stroop, 'continuous'),
    'yelp': ('beyondmlr/data/yelp.csv/2', yelp, 'binary'),
}


def export(name: str, cache: Path, out_dir: Path) -> None:
    rid, extract, y_type = SOURCES[name]
    registry = {r.id: r for r in rows()}
    x, y, names = extract(Tables(cache).load(registry[rid]))
    assert not x.isna().any().any() and not y.isna().any() and not names.isna().any()
    codes, levels = pd.factorize(names, sort=True)
    order = np.argsort(codes, kind='stable')
    groups = codes[order]
    path = out_dir / f'{name}.npz'
    np.savez_compressed(
        path,
        X=x.to_numpy(float)[order],
        y=y.to_numpy()[order],
        groups=groups,
        columns=np.array(x.columns, dtype=str),
        d=np.array(x.shape[1] + 1),
        n=np.array(len(groups)),
        ns=np.bincount(groups),
        m=np.array(len(levels)),
        y_type=np.array(y_type),
        group_names=np.array(levels, dtype=str),
        registry_id=np.array(rid),
    )
    print(f'{name}: {len(groups)} rows, {len(levels)} groups -> {path}')


def main() -> None:
    cache, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        export(name, cache, out_dir)


if __name__ == '__main__':
    main()
