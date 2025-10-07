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
