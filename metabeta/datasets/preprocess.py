"""Bulk preprocessing script for srm, pmlb, ucimlr, and from-r datasets.

Run directly to preprocess all datasets:
    python -m metabeta.datasets.preprocess
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metabeta.datasets.preprocessor import (
    preprocess,
    preprocessAllGroups,
)
from metabeta.plotting import plotDataset
from metabeta.utils.plot import DPI

logger = logging.getLogger(__name__)
DATASETS_DIR = Path(__file__).resolve().parent

TEST_GROUP_WHITELIST: set[tuple[str, str, str]] = {
    ('from-r', 'math', 'group'),
    ('from-r', 'london', 'group'),
    ('from-r', 'gcse', 'group'),
    ('from-r', 'sleep', 'group'),
    ('from-r', 'orthodont', 'group'),
    ('from-r', 'oxboys', 'group'),
    ('from-r', 'penicillin', 'group'),
    ('from-r', 'cbpp', 'group'),
    ('from-r', 'theoph', 'group'),
    ('from-r', 'orange', 'group'),
    ('from-r', 'indometh', 'group'),
    ('from-r', 'insteval', 'group'),
    ('from-r', 'verbagg', 'group'),
    ('from-r', 'grouseticks', 'group'),
    ('from-r', 'pastes', 'group'),
    ('from-r', 'ergostool', 'group'),
    ('from-r', 'machines', 'group'),
    ('from-r', 'oats', 'group'),
    ('from-r', 'pixel', 'group'),
    ('from-r', 'hsb82', 'group'),
    ('from-r', 'chem97', 'group'),
    ('from-r', 'contraception', 'group'),
    ('pmlb', '556_analcatdata_apnea2', 'subject'),
    ('pmlb', '557_analcatdata_apnea1', 'subject'),
    ('pmlb', 'analcatdata_boxing1', 'judge'),
    ('pmlb', 'analcatdata_boxing2', 'judge'),
}


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------


def processDataset(
    ds_name: str,
    root: str,
    group_name: str = '',
    partition: str = 'auto',
    save: bool = True,
    plot: bool = False,
):
    assert partition in ['validation', 'test', 'auto']

    fn = DATASETS_DIR / root / 'parquet' / f'{ds_name}.parquet'
    assert fn.exists(), f'File {fn} does not exist.'
    print(f'\nProcessing {ds_name}...')
    df = pd.read_parquet(fn)

    if len(df) > 100_000:
        logger.error('Dataset has more than 100k rows, skipping.')
        return

    if group_name:
        try:
            variants = {group_name: preprocess(df, group_name=group_name)}
        except ValueError as e:
            logger.error(f'Dataset "{ds_name}" skipped: {e}')
            return
    else:
        try:
            variants = preprocessAllGroups(df)
        except ValueError as e:
            logger.error(f'Dataset "{ds_name}" skipped: {e}')
            return

    out: dict[str, dict] = {}
    for grp_name, data in variants.items():
        if data['X'].shape[0] < data['X'].shape[1]:
            logger.error(
                f'"{ds_name}" (group="{grp_name or "none"}") more columns than rows, skipping.'
            )
            continue
        if data['n'] < 20:
            logger.warning(
                f'"{ds_name}" (group="{grp_name or "none"}") too few rows (n={data["n"]}); skipping.'
            )
            continue

        is_whitelisted = (root, ds_name, grp_name) in TEST_GROUP_WHITELIST
        if data['X'].shape[1] == 0:
            if is_whitelisted:
                logger.warning(
                    f'"{ds_name}" (group="{grp_name or "none"}") no predictors; keeping intercept-only.'
                )
            else:
                logger.warning(
                    f'"{ds_name}" (group="{grp_name or "none"}") no predictors; skipping.'
                )
                continue

        partition_i = partition
        if partition_i == 'auto':
            if data['groups'] is None:
                partition_i = 'validation'
            else:
                partition_i = 'test' if is_whitelisted else 'validation'

        suffix = ''
        if grp_name:
            safe_group = re.sub(r'[^0-9a-zA-Z_]+', '_', grp_name)
            suffix = f'__grp_{safe_group}'
        stem = f'{ds_name}{suffix}'

        if save:
            fn_out = DATASETS_DIR / 'preprocessed' / partition_i / f'{stem}.npz'
            np.savez_compressed(fn_out, **{k: v for k, v in data.items() if v is not None})
            print(f'Saved to {fn_out.relative_to(DATASETS_DIR)}')

        if plot and data['n'] * data['d'] < 1e6:
            dat = np.concatenate([data['y'][:, None], data['X']], axis=-1)
            names = ['y'] + data['columns'].tolist()
            fig = plotDataset(dat, names, kde=len(dat) < 10_000)
            if save:
                fn_plot = DATASETS_DIR / 'preprocessed' / 'plots' / f'{stem}.pdf'
                fig.savefig(fn_plot, dpi=DPI)
                plt.close()

        out[stem] = data

    return out


def batchProcess(root: str, group_name: str = '', partition: str = 'auto'):
    paths = Path(DATASETS_DIR, root, 'parquet').glob('*.parquet')
    names = sorted([p.stem for p in paths])
    logger.info(f'\nProcessing {len(names)} datasets from {root}...')
    for name in names:
        processDataset(name, root, group_name=group_name, partition=partition)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    (DATASETS_DIR / 'preprocessed' / 'validation').mkdir(parents=True, exist_ok=True)
    (DATASETS_DIR / 'preprocessed' / 'test').mkdir(parents=True, exist_ok=True)
    (DATASETS_DIR / 'preprocessed' / 'plots').mkdir(parents=True, exist_ok=True)

    batchProcess('from-r', partition='test', group_name='group')
    batchProcess('srm', partition='auto')
    batchProcess('pmlb', partition='auto')
    batchProcess('ucimlr', partition='auto')
