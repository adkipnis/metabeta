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
    def __init__(
        self,
        data: dict | None = None,
        path: Path | None = None,
        d: int = 5,
        q: int = 2,
        **kwargs,
    ):
        # load and store as tensors
        if data is None:
            assert path is not None, "must provide either data or path"
            assert path.exists(), (
                f"File {path} does not exist, you must generate it first using generate.py"
            )
            np_dat = np.load(path, allow_pickle=True)
            data = {k: np_dat[k] for k in np_dat.files}
            del np_dat

        # sizes
        self.d = d
        self.q = q
        self.len = data["y"].shape[0]
        self.max_n_i = int(max(data["n_i"]))

        # default priors
        self.nu_ffx = torch.zeros(self.d)
        self.tau_ffx = torch.ones(self.d) * 50.0
        self.tau_ffx[0] *= 2  # intercept prior
        self.tau_rfx = torch.ones(self.d) * 30.0
        self.tau_eps = torch.tensor(10.0)

        # make data compliant for model
        self.original = data
        data = {
            k: torch.from_numpy(v)
            for k, v in data.items()
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)
        }
        self.data_long = self.curate(data)
        self.data = self.deepen(self.data_long)

        # save all possible permutations for workers
        perms = torch.tensor(list(itertools.permutations(range(d - 1)))) + 1
        zeros = torch.zeros((len(perms), 1))
        self.perms = torch.cat([zeros, perms], dim=-1).int()
        self.unperms = torch.stack([inversePermutation(p) for p in self.perms])

    def __len__(self):
        return self.len

    def addIntercept(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(x[..., 0:1])
        return torch.cat([ones, x], dim=-1)

    def curate(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # sizes
        d = data["d"]
        self.d = int(max(d, self.d))
        q = torch.tensor(1)  # this should be dynamic
        n = data["n"]
        m = data["m"]
        n_i = data["n_i"]

        # inputs
        y = data["y"].float()
        X = data["X"]
        X = self.addIntercept(X).float()
        Z = X.clone()
        Z[..., q:] = 0
        groups = data["groups"]

        # mask priors
        self.tau_ffx[d:] = 0.0
        self.tau_rfx[q:] = 0.0

        # output
        out = {
            # sizes
            "m": m,
            "n": n,
            "n_i": n_i,
            "d": d,
            "q": q,
            # inputs
            "y": y,
            "X": X,
            "Z": Z,
            "groups": groups,
            # priors
            "nu_ffx": self.nu_ffx,
            "tau_ffx": self.tau_ffx,
            "tau_eps": self.tau_eps,
            "tau_rfx": self.tau_rfx,
        }
        return out

    def deepen(self, ds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ds = {k: v.clone() for k, v in ds.items()}
        y, X, Z = ds["y"], ds["X"], ds["Z"]
        n_i = ds["n_i"]
        m = int(ds["m"])
        d = ds["d"]
        q = ds["q"]

        y = split(y, n_i, shape=[m, self.max_n_i])
        X = split(X, n_i, shape=[m, self.max_n_i, self.d])
        Z = split(Z, n_i, shape=[m, self.max_n_i, self.d])

        # mask
        mask_d = torch.ones(self.d, dtype=torch.bool)
        mask_d[d:] = False
        mask_n = y != 0.0
        mask_m = torch.ones(m, dtype=torch.bool)
        mask_q = torch.ones(self.d, dtype=torch.bool)
        mask_q[q:] = False
        non_empty = torch.ones(self.d, dtype=torch.bool)
        non_empty[self.q :] = False

        out = {
            # inputs
            "y": y,
            "X": X,
            "Z": Z,
            # masks
            "mask_d": mask_d,
            "mask_q": mask_q,
            "mask_n": mask_n,
            "mask_m": mask_m,
            "non_empty": non_empty,
        }
        ds.update(**out)
        return ds

    def permute(
        self, ds: dict[str, torch.Tensor], perm: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        ds = {k: v.clone() for k, v in ds.items()}
        ds["X"] = ds["X"][..., perm]
        ds["Z"] = ds["Z"][..., perm]
        ds["nu_ffx"] = ds["nu_ffx"][..., perm]
        ds["tau_ffx"] = ds["tau_ffx"][..., perm]
        ds["tau_rfx"] = ds["tau_rfx"][..., perm]
        ds["mask_d"] = ds["mask_d"][..., perm]
        ds["mask_q"] = ds["mask_q"][..., perm]
        return ds

    def raw(self):
        return self.data_long

    def batch(self, n_workers: int = 8, device: str = "cpu") -> dict[str, torch.Tensor]:
        collate_fn = getCollater(device=device)
        if n_workers > len(self.perms):
            print(
                f"Only {len(self.perms)} unique permutations are possible, reducing n_workers."
            )
            n_workers = len(self.perms)
        out = []
        for perm, unperm in zip(self.perms[:n_workers], self.unperms[:n_workers]):
            ds = self.permute(self.data, perm)
            mask = ds["non_empty"][perm]
            ds["Z"] = ds["Z"][..., mask]
            ds["tau_rfx"] = ds["tau_rfx"][..., mask]
            ds["mask_q"] = ds["mask_q"][..., mask]
            ds["perm"] = perm
            ds["unperm"] = unperm
            out += [ds]
        return collate_fn(out)


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

    rds = RealDataset(raw, d=5, q=3)
    ds_long = rds.data_long
    ds = rds.batch()

