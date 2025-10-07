from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import torch
from metabeta.data.dataset import split, getCollater, inversePermutation

DATA_DIR = Path("real")


def removeString(this: list, string: str):
    out = list(filter(lambda x: x != string, this))
    return np.array(out)


def categorial(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return cat_cols


def numerical(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns
    return num_cols


def dummify(df: pd.DataFrame, colname: str):
    ref_category = df[colname].mode()[0]
    dummies = pd.get_dummies(df[colname], prefix=colname).astype(int)
    dummies = dummies.drop(f"{colname}_{ref_category}", axis=1)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(colname, axis=1)
    return df


def demean(df: pd.DataFrame, colname: str):
    df = df.copy()
    df[colname] -= df[colname].mean()
    return df


def preprocess(ds_name: str, target_name: str, group_name: str, save: bool = False):
    # import data
    fn = Path(DATA_DIR, f"{ds_name}.csv")
    assert fn.exists(), (
        f'File {fn} does not exist, have you generated it with "datasets.R"?'
    )
    df_orig = pd.read_csv(fn)
    df = df_orig.dropna()

    # sort and isolate grouping variable
    df.sort_values(by=group_name)
    groups = df.pop(group_name)
    groups, _ = pd.factorize(groups)
    _, n_i = np.unique(groups, return_counts=True)

    # isolate target
    y = df.pop(target_name).to_numpy()

    # mean-center numeric variables
    col_names_num = numerical(df)
    means = df[col_names_num].mean().to_numpy()  # type: ignore
    for n in col_names_num:
        df = demean(df, n)

    # dummy-code categorical variables
    col_names_cat = categorial(df)
    for c in col_names_cat:
        df = dummify(df, c)

    # finalize
    col_names_final = df.columns
    X = df.to_numpy()
    R = torch.corrcoef(torch.tensor(X).permute(1, 0))
    out = {
        "X": X,
        "y": y,
        "groups": groups,
        "y_name": target_name,
        "X_names": col_names_final,
        "numeric_names": col_names_num,
        "original_means": means,
        "n": np.array(len(df)),
        "n_i": n_i,
        "m": np.array(len(n_i)),
        "d": np.array(len(col_names_final) + 1),
        "cor": R,
    }

    # save
    if save:
        fn = Path(DATA_DIR, f"{ds_name}.npz")
        np.savez_compressed(fn, **out)
        print(f"Saved preprocessed dataset to {fn}.")

    return out


class RealDataset:
