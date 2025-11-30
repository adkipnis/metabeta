from pathlib import Path
import pandas as pd
import numpy as np


BLACKLIST = 'year age height size n_ num_ number max min attempts begin end name'.split(' ')
GREENLIST = 'country state school education class industry occupation race sport brand genre color weekday date'.split(' ')


def removeString(this: list, string: str):
    out = list(filter(lambda x: x != string, this))
    return np.array(out)


def dropConstantColumns(df: pd.DataFrame, threshold: float = 0.98) -> pd.DataFrame:
    # drop columns with at least {threshold} constant values
    bad = []
    for col in df.columns.drop('y'):
        counts = df[col].value_counts(normalize=True)
        if counts.max() > threshold:
            bad.append(col)
            print(f'--- Warning: Removing "{col}" due to mostly constant entries.')
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


def dropPatchyRows(df: pd.DataFrame):
    # drop rows with missing values
    df = df.replace(-999, None)
    missing = df.drop(columns='y').isnull().any(axis=1)
    if missing.mean() > 0.1:
        print(f'--- Warning: {missing.mean() * 100:.2f}% patchy rows.')
    df = df[~missing]
    return df


def potentialGroups(df: pd.DataFrame,
                    threshold: float = 0.2, minimum: int = 5):
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


def rescale(col: pd.Series):
    # scale down columns with very large values
    x = col.values.astype(float)
    max_abs = np.max(np.abs(x))
    scale_factor = 1.
    if max_abs > 1000:
        order = int(np.floor(np.log10(max_abs))) - 2
        scale_factor = 10. ** order
    return x / scale_factor


def conditionalCenter(col: pd.Series, threshold: float = 0.25):
    # center column if less than {threshold} of its entries are zero
    zero_count = (col == 0).sum()
    if zero_count < len(col) * threshold:
        col = col - col.mean()
    return col


def findOutliers(df: pd.DataFrame, threshold: float = 4.0):
    # detect values that are more than {threshold} SDs away from the mean
    mean = df.mean()
    std = df.std()
    z = (df - mean) / std
    outliers = (z.abs() > threshold).any(axis=1)
    if outliers.sum() / len(df) > 0.1:
        print(f'--- Warning: Removing {outliers.sum()} outliers.')
    return outliers


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


def preprocess(ds_name: str,
               root: str,
               group_name: str = '',
               target_name: str = 'y',
               partition: str = 'train',
               move_to_test: bool = True,
               save: bool = True):
    # import data
    fn = Path(root, 'parquet', f'{ds_name}.parquet')
    assert fn.exists(), f'File {fn} does not exist.'
    df_orig = pd.read_parquet(fn)

    # discard gigantic datasets
    if len(df_orig) > 500_000:
        print('--- Fatal: Dataset has more than 500k rows, skipping.')
        return

    # put column names to lower case
    df_orig.columns = df_orig.columns.str.lower()
    df = df_orig

    # remove columns with more than 98% constant values
    df = dropConstantColumns(df)

    # remove columns with more than 25% missing values
    df = dropPatchyColumns(df)

    # remove rows with missing values
    df = dropPatchyRows(df)

    # isolate target
    y = df.pop(target_name).to_numpy()

    # detect potential grouping variables
    if not group_name:
        potential = potentialGroups(df)
        potential = [p for p in potential if p in GREENLIST]
        if len(potential):
            group_name = potential[0]
            print(f'--- Note: Detected grouping variable "{group_name}".')
            if len(potential) > 1:
                print(f'--- Warning: Removing other grouping variables {potential[1:]}.')
                for p in potential[1:]:
                    df.pop(p)
            if move_to_test:
                partition = 'test'

    # sort and isolate grouping variable
    groups = n_i = m = None
    if group_name:
        df.sort_values(by=group_name)
        groups = df.pop(group_name)
        groups, _ = pd.factorize(groups)

    # analyze column types
    col_names_num = numerical(df)
    col_names_cat = categorical(df)

    # remove outliers
    outliers = findOutliers(df[col_names_num])
    df = df[~outliers]
    y = y[~outliers]

    # optionally update group objects
    if groups is not None:
        groups = groups[~outliers]
        _, n_i = np.unique(groups, return_counts=True)
        m = n_i.shape[0]

    # scale down
    df[col_names_num] = df[col_names_num].apply(rescale)

    # demean
    means = df[col_names_num].mean().to_numpy()  # type: ignore
    df[col_names_num] = df[col_names_num].apply(conditionalCenter)

    # dummy-code categorical variables
    for col in col_names_cat:
        df = dummify(df, col)

    # discard datasets that are wider than long
    if df.shape[0] < df.shape[1]:
        print('--- Fatal: Dataset has more columns than rows, skipping.')
        return

    # finalize
    col_names_final = df.columns
    X = df.to_numpy()
    n, d = X.shape
    cor = np.corrcoef(X, rowvar=False)
    out = {
        # data
        'X': X,
        'y': y,
        'groups': groups,
        'cor': cor,
        # names
        'columns': col_names_final,
        'numeric': col_names_num,
        'means': means,
        # dims
        'd': d,
        'n': n,
        'n_i': n_i,
        'm': m,
    }

    # save
    if save:
        fn = Path('preprocessed', partition, f'{ds_name}.npz')
        np.savez_compressed(fn, **out)
        print(f'Saved to {fn}')

    return out


def batchprocess(root: str, group_name: str = '', partition: str = 'train'):
    paths = Path(root, 'parquet').glob("*.parquet")
    names = sorted([p.stem for p in paths])
    print(f'\nProcessing {len(names)} datasets from {root}...')
    for name in names:
        preprocess(name, root, group_name=group_name, partition=partition)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # init preprocessed directory
    Path('preprocessed', 'train').mkdir(parents=True, exist_ok=True)
    Path('preprocessed', 'test').mkdir(parents=True, exist_ok=True)

    # r-package datasets
    batchprocess('from-r', partition='test', group_name='group')

    # srm datasets
    batchprocess('srm')

    # # pmlb datasets
    # batchprocess('pmlb')

    # # automl datasets (skipped, too many issues)
    # batchprocess('automl', partition='train')

