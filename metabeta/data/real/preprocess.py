from pathlib import Path
import pandas as pd
import numpy as np
import torch


def removeString(this: list, string: str):
    out = list(filter(lambda x: x != string, this))
    return np.array(out)


def categorical(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    return cat_cols


def numerical(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return num_cols

def rescale(df: pd.DataFrame, colname: str):
    x = df[colname].values.astype(float)
    max_abs = np.max(np.abs(x))
    scale_factor = 1    
    if max_abs > 1000:
        order = int(np.floor(np.log10(max_abs))) - 2
        scale_factor = 10 ** order
    df[colname] = x / scale_factor
    return df
    

def dummify(df: pd.DataFrame, colname: str):
    ref_category = df[colname].mode()[0]
    dummies = pd.get_dummies(df[colname], prefix=colname).astype(int)
    dummies = dummies.drop(f'{colname}_{ref_category}', axis=1)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(colname, axis=1)
    return df


def demean(df: pd.DataFrame, colname: str):
    df = df.copy()
    df[colname] -= df[colname].mean()
    return df


def preprocess(ds_name: str,
               target_name: str,
               group_name: str = '',
               save: bool = False):
    # import data
    fn = Path(root, f'{ds_name}.csv')
    assert fn.exists(), f'File {fn} does not exist.'
    df_orig = pd.read_csv(fn)

    # remove missing values
    # TODO: check for offending columns and optionally remove them first
    df = df_orig.dropna()

    # sort and isolate grouping variable
    if group_name:
        df.sort_values(by=group_name)
        groups = df.pop(group_name)
        groups, _ = pd.factorize(groups)
        _, n_i = np.unique(groups, return_counts=True)
        m = n_i.shape[0]
    else:
        groups = None
        n_i = None
        m = None

    # isolate target
    y = df.pop(target_name).to_numpy()

    # mean-center numeric variables
    col_names_num = numerical(df)
    means = df[col_names_num].mean().to_numpy()  # type: ignore
    for n in col_names_num:
        df = demean(df, n)

    # dummy-code categorical variables
    col_names_cat = categorical(df)
    for c in col_names_cat:
        df = dummify(df, c)

    # finalize
    col_names_final = df.columns
    X = df.to_numpy()
    n, d = X.shape
    R = torch.corrcoef(torch.tensor(X).permute(1, 0))
    out = {
        # data
        'X': X,
        'y': y,
        'groups': groups,
        'cor': R,
        # names
        'y_name': target_name,
        'X_names': col_names_final,
        'numeric_names': col_names_num,
        'original_means': means,
        # dims
        'd': d,
        'n': n,
        'n_i': n_i,
        'm': m,
    }

    # save
    if save:
        fn = Path('preprocessed', f'{ds_name}.npz')
        np.savez_compressed(fn, **out)
        print(f'Saved preprocessed dataset to {fn}')

    return out

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    raw = preprocess(
            ds_name = 'math',
            target_name = 'MathAch',
            group_name = 'School',
            save = True
            )

    raw = preprocess(
            ds_name = 'exam',
            target_name = 'normexam',
            group_name = 'school',
            save = True
            )

    raw = preprocess(
            ds_name = 'gcsemv',
            target_name = 'written',
            group_name = 'school',
            save = True
            )

    raw = preprocess(
            ds_name = 'sleepstudy',
            target_name = 'Reaction',
            group_name = 'Subject',
            save = True
            )


