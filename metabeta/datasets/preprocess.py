from pathlib import Path
import pandas as pd
import numpy as np
from metabeta.plot import plot
from matplotlib import pyplot as plt

BLACKLIST = 'year age height size n_ num_ number max min attempts begin end name'.split(' ')
GREENLIST = 'country state school education class industry occupation race sport brand genre color weekday date'.split(' ')
PARAMS = []


def removeString(this: list, string: str) -> np.ndarray:
    out = list(filter(lambda x: x != string, this))
    return np.array(out)


def dropConstantColumns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    # drop columns with at least {threshold} constant values
    bad = []
    for col in df.columns:
        counts = df[col].value_counts(normalize=True)
        if counts.max() > threshold:
            bad.append(col)
    if len(bad):
        print(f'--- Warning: Removing {bad} due to mostly constant entries.')
    return df.drop(columns=bad)


def dropPatchyColumns(df: pd.DataFrame, threshold: float = 0.25):
    # drop columns with at least {threshold} missing values
    missing = df.drop(columns='y').isnull().mean()
    offenders = missing[missing > threshold].index
    if len(offenders):
        for offender in offenders:
            print(f'--- Warning: Removing "{offender}" due to missing {missing[offender]*100:.2f}% entries.')
        df = df.drop(columns=offenders)
    return df


def dropPatchyRows(df: pd.DataFrame, threshold: float = 0.1):
    # drop rows with missing values
    df = df.replace(-999, np.nan)
    missing = df.drop(columns='y').isnull().any(axis=1)
    if missing.mean() > threshold:
        print(f'--- Warning: {missing.mean() * 100:.2f}% patchy rows.')
    df = df[~missing]
    return df


def potentialGroups(
    df: pd.DataFrame,
    threshold: float = 0.2,
    minimum: int = 5):
    # detect columns that may contain a grouping variable
    assert 0. < threshold < 1., 'threshold must be in (0,1)'

    # get non-continuous columns
    cols = df.select_dtypes(exclude=['float']).columns

    # exclude blacklisted columns
    blacklisted = cols.str.contains('|'.join(BLACKLIST))
    cols = cols[~blacklisted]

    # quick escape
    if len(cols) == 0:
        return cols

    # check if a group has enough but not too many unique values
    counts = df[cols].nunique()
    enough = (counts > minimum)
    not_too_many = (counts <= threshold * len(df))
    mask = (enough * not_too_many)
    cols = cols[mask]
    if len(cols) == 0:
        return cols

    # check if group sizes are somewhat balanced
    ratios = np.zeros(len(cols))
    for i, col in enumerate(cols):
        group_counts = df[col].value_counts()
        ratios[i] = group_counts.max() / group_counts.min()
    mask = pd.Series(ratios < 10)
    cols = cols[mask]
    return cols


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
        print(f'--- Warning: Removing {frac * 100:.2f}% outlier rows.')
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

def dummify(df: pd.DataFrame, colname: str, max_columns: int = 10):
    # make dummy variables out of categorical columns
    unique = df[colname].nunique()
    if unique > max_columns:
        print(f'--- Warning: Removing category {colname} due to {unique} factors.')
        return df.drop(colname, axis=1)
    ref_category = df[colname].mode()[0]
    dummies = pd.get_dummies(df[colname], prefix=colname).astype(int)
    dummies = dummies.drop(f'{colname}_{ref_category}', axis=1)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(colname, axis=1)
    return df


def preprocess(df: pd.DataFrame,
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
    y = df.pop('y')

    # detect potential grouping variables
    if not group_name:
        potential = potentialGroups(df)
        if len(potential):
            group_name = potential[0]
            print(f'--- Note: Detected grouping variable "{group_name}".')
            if len(potential) > 1:
                print(f'--- Warning: Removing other grouping variables {potential[1:]}.')
                for p in potential[1:]:
                    df.pop(p)

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
        groups = groups[~outliers]
        groups, _ = pd.factorize(groups)
        _, ns = np.unique(groups, return_counts=True)
        m = len(ns)

    # remove columns with more than 95% constant values
    df = dropConstantColumns(df, threshold=constant_threshold)

    # (re-)analyze column types
    col_names_num = numerical(df)
    col_names_cat = categorical(df)

    # z-standardize
    df[col_names_num] = df[col_names_num].apply(standardize)
    y = standardize(y)

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
        'd': d_+1, # as X has no intercept
        'n': n,
        'ns': ns,
        'm': m,
    }
    return out


def wrapper(ds_name: str,
            root: str,
            group_name: str = '',
            partition: str = 'auto',
            save: bool = True,
            plot_ds: bool = False):
    assert partition in ['validation', 'test', 'auto']

    # import data
    fn = Path(root, 'parquet', f'{ds_name}.parquet')
    assert fn.exists(), f'File {fn} does not exist.'
    print(f'\nProcessing {ds_name}...')
    df = pd.read_parquet(fn)

    # discard gigantic datasets
    if len(df) > 100_000:
        print('--- Fatal: Dataset has more than 100k rows, skipping.')
        return

    # preprocess dataset
    data = preprocess(df, group_name=group_name)

    # discard datasets that are wider than long
    if data['X'].shape[0] < data['X'].shape[1]:
        print('--- Fatal: Dataset has more columns than rows, skipping.')
        return

    # determine partition
    if partition == 'auto':
        partition = 'validation' if data['groups'] is None else 'test'

    # save
    if save:
        fn = Path('preprocessed', partition, f'{ds_name}.npz')
        np.savez_compressed(fn, **data)
        print(f'Saved to {fn}')

    # plot
    if plot_ds and np.prod(df.shape) < 1e6:
        dat = np.concatenate([data['y'][:, None], data['X']], axis=-1)
        names = ['y'] + data['columns'].tolist()
        fig = plot.dataset(dat, names, kde=len(dat) < 10_000)
        if save:
            fn = Path('preprocessed', 'plots', f'{ds_name}.pdf')
            fig.savefig(fn, dpi=300)
            plt.close()
    return data


def batchprocess(root: str, group_name: str = '', partition: str = 'auto'):
    paths = Path(root, 'parquet').glob("*.parquet")
    names = sorted([p.stem for p in paths])
    print(f'\nProcessing {len(names)} datasets from {root}...')
    for name in names:
        wrapper(name, root, group_name=group_name, partition=partition)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # init preprocessed directory
    Path('preprocessed', 'validation').mkdir(parents=True, exist_ok=True)
    Path('preprocessed', 'test').mkdir(parents=True, exist_ok=True)
    Path('preprocessed', 'plots').mkdir(parents=True, exist_ok=True)

    # r-package datasets
    batchprocess('from-r', partition='test', group_name='group')

    # srm datasets
    batchprocess('srm')

    # pmlb datasets
    batchprocess('pmlb')

    # # automl datasets (skipped, too many issues)
    # batchprocess('automl', partition='train')

