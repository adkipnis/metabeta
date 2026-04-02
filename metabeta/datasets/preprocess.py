import logging
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from metabeta.plotting import plotDataset
from metabeta.utils.plot import DPI
from metabeta.utils.preprocessing import transformPredictors

logger = logging.getLogger(__name__)
DATASETS_DIR = Path(__file__).resolve().parent

BLACKLIST = 'year age height size n_ num_ number max min attempts begin end name'.split(' ')

# Grouped variants in this whitelist are routed to the test partition when partition='auto'.
# All other grouped variants are routed to validation.
TEST_GROUP_WHITELIST: set[tuple[str, str, str]] = {
    # handpicked/curated from-r exports
    ('from-r', 'math', 'group'),
    ('from-r', 'london', 'group'),
    ('from-r', 'gcse', 'group'),
    ('from-r', 'sleep', 'group'),
    ('from-r', 'dyestuff', 'group'),
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
    ('from-r', 'rail', 'group'),
    ('from-r', 'ergostool', 'group'),
    ('from-r', 'machines', 'group'),
    ('from-r', 'oats', 'group'),
    ('from-r', 'pixel', 'group'),
    ('from-r', 'hsb82', 'group'),
    ('from-r', 'chem97', 'group'),
    ('from-r', 'contraception', 'group'),
    # clearly plausible mixed-effects datasets in PMLB
    ('pmlb', '556_analcatdata_apnea2', 'subject'),
    ('pmlb', '557_analcatdata_apnea1', 'subject'),
    ('pmlb', 'analcatdata_boxing1', 'judge'),
    ('pmlb', 'analcatdata_boxing2', 'judge'),
}
MAX_GROUP_CANDIDATES = 3


@dataclass(frozen=True)
class GroupCandidate:
    name: str
    n_groups: int
    frac_unique: float
    avg_obs_per_group: float
    min_obs_per_group: int
    max_obs_per_group: int
    imbalance: float
    score: float


def dropPatchyColumns(df: pd.DataFrame, threshold: float = 0.25):
    # drop columns with at least {threshold} missing values
    missing = df.drop(columns='y').isnull().mean()
    offenders = missing[missing > threshold].index
    if len(offenders):
        for offender in offenders:
            warning = (
                f'Removing "{offender}" due to missing {missing[offender] * 100:.2f}% entries.'
            )
            logger.warning(warning)
        df = df.drop(columns=offenders)
    return df


def dropPatchyRows(df: pd.DataFrame, threshold: float = 0.1, sentinels: list[float] | None = None):
    # drop rows with missing values
    if sentinels is None:
        sentinels = [-999]
    for s in sentinels:
        df = df.replace(s, np.nan)
    missing = df.drop(columns='y').isnull().any(axis=1)
    if missing.mean() > threshold:
        logger.warning(f'{missing.mean() * 100:.2f}% patchy rows.')
    df = df[~missing]
    return df


def detectGroupCandidates(
    df: pd.DataFrame,
    min_groups: int = 5,
    max_frac_unique: float = 0.5,
    min_obs_per_group: int = 2,
    max_groups_cap: int = 200,
    min_frac_singleton: float = 0.05,
    max_frac_singleton: float = 0.35,
) -> list[GroupCandidate]:
    assert 0.0 < max_frac_unique < 1.0, 'max_frac_unique must be in (0,1)'

    if len(df) == 0:
        return []

    eligible: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            vals = s.dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            if not np.allclose(vals, np.round(vals), atol=1e-12, rtol=0):
                continue
        elif not (
            pd.api.types.is_integer_dtype(s)
            or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        ):
            continue
        eligible.append(col)

    if not eligible:
        return []

    pattern = '|'.join(rf'\b{word}\b' for word in BLACKLIST)
    eligible = [col for col in eligible if re.search(pattern, col) is None]
    if not eligible:
        return []

    n = len(df)
    max_groups = min(max_groups_cap, int(max_frac_unique * n))
    candidates: list[GroupCandidate] = []
    for col in eligible:
        counts = df[col].value_counts(dropna=True)
        if len(counts) == 0:
            continue

        n_groups = int(len(counts))
        if n_groups < min_groups or n_groups > max_groups:
            continue

        min_count = int(counts.min())
        if min_count < min_obs_per_group:
            continue

        max_count = int(counts.max())
        avg_count = float(counts.mean())
        frac_unique = float(n_groups / n)
        imbalance = float(max_count / max(min_count, 1))
        frac_singleton = float(np.mean(counts == 1))

        score = 0.0
        score -= abs(np.log10(max(avg_count, 1e-12)) - np.log10(20.0))
        if 8 <= n_groups <= 150:
            score += 0.5
        score -= 0.1 * max(0.0, np.log10(max(imbalance, 1.0)))
        score -= 1.0 * abs(frac_singleton - 0.15)

        lower = col.lower()
        if re.search(r'group|subject|school|cluster|site|center|class|id', lower):
            score += 0.5
        if re.search(r'train|test|valid|fold|split', lower):
            score -= 1.0
        if frac_singleton < min_frac_singleton or frac_singleton > max_frac_singleton:
            score -= 0.75

        candidates.append(
            GroupCandidate(
                name=col,
                n_groups=n_groups,
                frac_unique=frac_unique,
                avg_obs_per_group=avg_count,
                min_obs_per_group=min_count,
                max_obs_per_group=max_count,
                imbalance=imbalance,
                score=score,
            )
        )

    return sorted(candidates, key=lambda c: (-c.score, c.name))


def categorical(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    return cat_cols


def numerical(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return num_cols


def standardize(col: pd.Series):
    x = col.values.astype(float)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std < 1e-12:
        return np.zeros_like(x)
    out = (x - mean) / std
    return out


def coerceTargetToNumeric(y: pd.Series) -> tuple[np.ndarray, bool]:
    if len(y) == 0:
        raise ValueError('Target y is empty.')

    if pd.api.types.is_numeric_dtype(y):
        arr = y.to_numpy(dtype=float)
        uniq = np.unique(arr[np.isfinite(arr)])
        is_binary = len(uniq) == 2 and set(uniq.tolist()).issubset({0.0, 1.0})
        return arr, is_binary

    y_str = y.astype('string').str.strip()
    non_missing = y_str.dropna()
    unique = sorted(non_missing.unique().tolist())

    if len(unique) == 2:
        lower = [u.lower() for u in unique]
        if set(lower) == {'n', 'y'}:
            mapper = {unique[lower.index('n')]: 0.0, unique[lower.index('y')]: 1.0}
        elif set(lower) == {'no', 'yes'}:
            mapper = {unique[lower.index('no')]: 0.0, unique[lower.index('yes')]: 1.0}
        elif set(lower) == {'false', 'true'}:
            mapper = {
                unique[lower.index('false')]: 0.0,
                unique[lower.index('true')]: 1.0,
            }
        else:
            mapper = {unique[0]: 0.0, unique[1]: 1.0}
        mapped = y_str.map(mapper).to_numpy(dtype=float)
        return mapped, True

    parsed = pd.to_numeric(y_str, errors='coerce')
    if parsed.notna().sum() == non_missing.shape[0]:
        arr = parsed.to_numpy(dtype=float)
        uniq = np.unique(arr[np.isfinite(arr)])
        is_binary = len(uniq) == 2 and set(uniq.tolist()).issubset({0.0, 1.0})
        return arr, is_binary

    sample = ', '.join(map(str, unique[:8]))
    raise ValueError(
        f'Cannot convert target y to numeric dtype. dtype={y.dtype}, sample values=[{sample}]'
    )


def findOutliers(df: pd.DataFrame, threshold: float = 4.0, min_std: float = 1e-12):
    # detect values that are more than {threshold} SDs away from the mean
    x = df.to_numpy(dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.maximum(std, min_std)
    z = (x - mean) / std
    outliers = (np.abs(z) > threshold).any(axis=1)
    frac = outliers.mean()
    if frac > 0.1:
        logger.warning(f'Removing {frac * 100:.2f}% outlier rows.')
    return outliers


# def findOutliersMAD(
#     df: pd.DataFrame,
#     threshold: float = 12.0,
#     min_std: float = 1e-12,
# ):
#     # MAD version of the above
#     x = df.to_numpy(dtype=float)
#     median = np.nanmedian(x, axis=0)
#     mad = np.nanmedian(np.abs(x - median), axis=0)
#     mad = np.maximum(mad, min_std)
#     z = 0.6745 * (x - median) / mad
#     outliers = (np.abs(z) > threshold).any(axis=1)
#     frac = outliers.mean()
#     if frac > 0.1:
#         print(f'--- Warning: Removing {frac * 100:.2f}% outlier rows.')
#     return outliers


def dummify(df: pd.DataFrame, colname: str, max_columns: int = 10, min_prevalence: float = 0.05):
    # make dummy variables out of categorical columns
    unique = df[colname].nunique()
    if unique > max_columns:
        logger.warning(f'Removing category {colname} due to {unique} factors.')
        return df.drop(colname, axis=1)
    ref_category = df[colname].mode()[0]
    dummies = pd.get_dummies(df[colname], prefix=colname).astype(int)
    dummies = dummies.drop(f'{colname}_{ref_category}', axis=1)

    # drop rare dummies
    rare = dummies.columns[dummies.mean() < min_prevalence]
    if len(rare):
        logger.warning(f'Removing rare dummies {list(rare)} (prevalence < {min_prevalence}).')
        dummies = dummies.drop(columns=rare)

    df = pd.concat([df, dummies], axis=1)
    df = df.drop(colname, axis=1)
    return df


def preprocess(
    df: pd.DataFrame,
    group_name: str = '',
    remove_missing: bool = True,
    patchy_threshold: float = 0.25,
    constant_threshold: float = 0.95,
    outlier_threshold: float = 4.0,
) -> dict:
    # put column names to lower case
    df.columns = df.columns.str.lower()

    # remove NA values
    if remove_missing:
        # remove columns with more than 25% missing values
        df = dropPatchyColumns(df, threshold=patchy_threshold)

        # remove rows with missing values
        df = dropPatchyRows(df)

    # isolate target
    assert 'y' in df.columns, 'target column y not present'
    y_raw = df.pop('y')
    y, y_is_binary = coerceTargetToNumeric(y_raw)

    # remove rows with missing/non-finite targets
    y_valid = np.isfinite(y)
    if not np.all(y_valid):
        n_bad = int((~y_valid).sum())
        logger.warning(f'Removing {n_bad} rows with missing/non-finite target values.')
        df = df.loc[y_valid].copy()
        y = y[y_valid]
        if len(df) == 0:
            raise ValueError(
                'All observations were removed due to missing/non-finite target values.'
            )

    # detect potential grouping variable
    if not group_name:
        candidates = detectGroupCandidates(df)
        if candidates:
            group_name = candidates[0].name
            logger.info(f'Detected grouping variable "{group_name}".')

    # sort and isolate grouping variable
    groups = ns = m = None
    if group_name:
        df = df.sort_values(by=group_name)
        groups = df.pop(group_name)
        groups, _ = pd.factorize(groups)

    # remove outliers
    col_names_num = numerical(df)
    outliers = findOutliers(df[col_names_num], threshold=outlier_threshold)
    df = df[~outliers]
    y = y[~outliers]

    # optionally update group objects
    if groups is not None:
        m_before = len(np.unique(groups))
        groups = groups[~outliers]
        groups, _ = pd.factorize(groups)
        _, ns = np.unique(groups, return_counts=True)
        m = len(ns)
        if m < m_before:
            logger.warning(f'{m_before - m} groups lost all observations after outlier removal.')

    # remove numeric columns with more than 95% constant values
    # (categorical near-constants are handled by dummify's min_prevalence)
    num_cols = numerical(df)
    bad = [c for c in num_cols if df[c].value_counts(normalize=True).max() > constant_threshold]
    if bad:
        logger.warning(f'Removing {bad} due to mostly constant entries.')
        df = df.drop(columns=bad)

    # (re-)analyze column types
    col_names_num = numerical(df)
    col_names_cat = categorical(df)

    # type-aware transform: log1p(count-like) + z-standardize (binary unchanged)
    if len(col_names_num):
        x_num = df[col_names_num].to_numpy(dtype=float)
        x_num = transformPredictors(x_num, axis=0, exclude_binary=True, transform_counts=True)
        df[col_names_num] = x_num
    if y_is_binary:
        y = y.astype(float)
    else:
        y = standardize(pd.Series(y))

    # dummy-code categorical variables
    for col in col_names_cat:
        df = dummify(df, col)

    # finalize
    X = df.to_numpy()
    n, d_ = X.shape
    out = {
        # data
        'X': X,
        'y': y,
        'groups': groups,
        # names
        'columns': df.columns.to_numpy().astype(str),
        # dims
        'd': d_ + 1,  # as X has no intercept
        'n': n,
        'ns': ns,
        'm': m,
    }
    return out


def preprocessAllGroups(
    df: pd.DataFrame,
    remove_missing: bool = True,
    patchy_threshold: float = 0.25,
    constant_threshold: float = 0.95,
    outlier_threshold: float = 4.0,
) -> dict[str, dict]:
    df = df.copy()
    df.columns = df.columns.str.lower()

    if remove_missing:
        df = dropPatchyColumns(df, threshold=patchy_threshold)
        df = dropPatchyRows(df)

    assert 'y' in df.columns, 'target column y not present'
    candidates = detectGroupCandidates(df.drop(columns='y'))
    if len(candidates) > MAX_GROUP_CANDIDATES:
        logger.warning(
            f'Found {len(candidates)} potential grouping variables; keeping top {MAX_GROUP_CANDIDATES} by score.'
        )
        candidates = candidates[:MAX_GROUP_CANDIDATES]
    out: dict[str, dict] = {}
    if not candidates:
        out[''] = preprocess(
            df.copy(),
            group_name='',
            remove_missing=False,
            patchy_threshold=patchy_threshold,
            constant_threshold=constant_threshold,
            outlier_threshold=outlier_threshold,
        )
        return out

    for cand in candidates:
        data = preprocess(
            df.copy(),
            group_name=cand.name,
            remove_missing=False,
            patchy_threshold=patchy_threshold,
            constant_threshold=constant_threshold,
            outlier_threshold=outlier_threshold,
        )
        out[cand.name] = data
    return out


def wrapper(
    ds_name: str,
    root: str,
    group_name: str = '',
    partition: str = 'auto',
    save: bool = True,
    plot: bool = False,
):
    assert partition in ['validation', 'test', 'auto']

    # import data
    fn = DATASETS_DIR / root / 'parquet' / f'{ds_name}.parquet'
    assert fn.exists(), f'File {fn} does not exist.'
    print(f'\nProcessing {ds_name}...')
    df = pd.read_parquet(fn)

    # discard gigantic datasets
    if len(df) > 100_000:
        logger.error('Dataset has more than 100k rows, skipping.')
        return

    if group_name:
        try:
            variants = {
                group_name: preprocess(df, group_name=group_name),
            }
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
                f'Dataset variant "{ds_name}" (group="{grp_name or "none"}") has more columns than rows, skipping.'
            )
            continue
        if data['n'] < 20:
            logger.warning(
                f'Dataset variant "{ds_name}" (group="{grp_name or "none"}") has too few rows after preprocessing (n={data["n"]}); skipping.'
            )
            continue

        is_whitelisted_grouped = (root, ds_name, grp_name) in TEST_GROUP_WHITELIST
        if data['X'].shape[1] == 0:
            if is_whitelisted_grouped:
                logger.warning(
                    f'Dataset variant "{ds_name}" (group="{grp_name or "none"}") has no predictors after preprocessing; keeping intercept-only design.'
                )
            else:
                logger.warning(
                    f'Dataset variant "{ds_name}" (group="{grp_name or "none"}") has no predictors after preprocessing; skipping.'
                )
                continue

        partition_i = partition
        if partition_i == 'auto':
            if data['groups'] is None:
                partition_i = 'validation'
            else:
                partition_i = 'test' if is_whitelisted_grouped else 'validation'

        suffix = ''
        if grp_name:
            safe_group = re.sub(r'[^0-9a-zA-Z_]+', '_', grp_name)
            suffix = f'__grp_{safe_group}'
        stem = f'{ds_name}{suffix}'

        if save:
            fn = DATASETS_DIR / 'preprocessed' / partition_i / f'{stem}.npz'
            np.savez_compressed(fn, **data)
            print(f'Saved to {fn.relative_to(DATASETS_DIR)}')

        if plot and data['n'] * data['d'] < 1e6:
            dat = np.concatenate([data['y'][:, None], data['X']], axis=-1)
            names = ['y'] + data['columns'].tolist()
            fig = plotDataset(dat, names, kde=len(dat) < 10_000)
            if save:
                fn = DATASETS_DIR / 'preprocessed' / 'plots' / f'{stem}.pdf'
                fig.savefig(fn, dpi=DPI)
                plt.close()

        out[stem] = data

    return out


def batchprocess(root: str, group_name: str = '', partition: str = 'auto'):
    paths = Path(root, 'parquet').glob('*.parquet')
    names = sorted([p.stem for p in paths])
    logger.info(f'\nProcessing {len(names)} datasets from {root}...')
    for name in names:
        wrapper(name, root, group_name=group_name, partition=partition)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # init preprocessed directory
    (DATASETS_DIR / 'preprocessed' / 'validation').mkdir(parents=True, exist_ok=True)
    (DATASETS_DIR / 'preprocessed' / 'test').mkdir(parents=True, exist_ok=True)
    (DATASETS_DIR / 'preprocessed' / 'plots').mkdir(parents=True, exist_ok=True)

    # R-package datasets
    batchprocess('from-r', partition='test', group_name='group')

    # SRM datasets
    batchprocess('srm', partition='auto')

    # PMLB datasets
    batchprocess('pmlb', partition='auto')

    # AutoML datasets
    # batchprocess('automl', partition='auto')
