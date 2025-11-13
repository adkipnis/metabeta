import os
from sys import exit
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch import distributions as D

from metabeta.utils import dsFilename, padTensor
from metabeta.data.tasks import MixedEffects
from metabeta.data.markov import fitMCMC
from metabeta.data.csv import RealDataset


# -----------------------------------------------------------------------------
# config
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate datasets for linear model task."
    )
    parser.add_argument("-t", "--type", type=str, default="mfx", help="Type of dataset (default = mfx)")
    parser.add_argument("--bs_train", type=int, default=4096, help="batch size per training partition (default = 4,096).")
    parser.add_argument("--bs_val", type=int, default=256, help="batch size for validation partition (default = 256).")
    parser.add_argument("--bs_test", type=int, default=128, help="batch size per testing partition (default = 128).")
    parser.add_argument("--bs_load", type=int, default=32, help="Batch size when loading (for grouping n, default = 32)")
    parser.add_argument("--min_m", type=int, default=5, help="MFX: Minimum number of groups (default = 5).")
    parser.add_argument("--max_m", type=int, default=30, help="MFX: Maximum number of groups (default = 30).")
    parser.add_argument("--min_n", type=int, default=10, help="Minimum number of samples per group (default = 10).")
    parser.add_argument("--max_n", type=int, default=70, help="Maximum number of samples per group (default = 70).")
    parser.add_argument("--max_d", type=int, default=4, help="Maximum number of fixed effects (intercept + slopes) to draw per linear model (default = 12).")
    parser.add_argument("--max_q", type=int, default=1, help="Maximum number of random effects (intercept + slopes) to draw per linear model (default = 4).")
    parser.add_argument("--d_tag", type=str, default="", help='Suffix for model ID (default = "")')
    parser.add_argument("--toy", action="store_false", help="Generate toy data (default = False)")
    parser.add_argument("--mono", action="store_true", help="Single prior per parameter type (default = False)")
    parser.add_argument("--semi", action="store_true", help="Generate semi-synthetic data (default = False)")
    parser.add_argument("--varied", action="store_true", help="variable d and q (default = False)")
    parser.add_argument("-b", "--begin", type=int, default=-1, help="Begin with iteration number #b (default = 0).")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of dataset partitions to generate (default = 100, 0 only generates validation dataset).")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# helpers
def getFileName(ds_type: str, part: int) -> Path:
    if "train" in ds_type:
        size = cfg.bs_train
        b = cfg.bs_load
    elif "val" in ds_type:
        size = cfg.bs_val
        b = 1
    elif "test" in ds_type:
        size = cfg.bs_test
        b = 1
    else:
        raise ValueError

    fn = dsFilename(
        fx_type=cfg.type,
        ds_type=ds_type,
        b=b,
        m=cfg.max_m if cfg.type == "mfx" else 0,
        n=cfg.max_n,
        d=cfg.max_d,
        q=cfg.max_q if cfg.type == "mfx" else 0,
        size=size,
        part=part,
        varied=cfg.varied,
        tag=cfg.d_tag,
        outside=True,
    )
    return fn


def sampleInt(batch_size: int | tuple, min_val: int, max_val: int) -> torch.Tensor:
    """sample batch from discrete uniform
    (include the max_val as possible value)"""
    shape = (batch_size,) if isinstance(batch_size, int) else batch_size
    return torch.randint(min_val, max_val + 1, shape)


def sampleIntBatched(
    min_val: int | torch.Tensor, max_val: int | torch.Tensor
) -> torch.Tensor:
    """batched version of the above with varying bounds"""
    if isinstance(min_val, int):
        min_val = torch.tensor([min_val])
    if isinstance(max_val, int):
        max_val = torch.tensor([max_val])
    n = max(len(min_val), len(max_val))
    samples = min_val + torch.floor((max_val + 1 - min_val) * torch.rand(n)).long()
    return samples


def sampleHN(shape: tuple, scale: float, max_val: float = float("inf")) -> torch.Tensor:
    """sample batch from halfnormal, optionally clamp"""
    val = D.HalfNormal(scale).sample(shape)
    if max_val < float("inf"):
        val = val.clamp(max=max_val) # type: ignore
    return val # type: ignore


# -----------------------------------------------------------------------------
# aggregation

def maxShapes(data: list[dict[str, torch.Tensor]]) -> dict[str, tuple[int, ...]]:
    ''' get the maximal shapes over the whole list for per key '''
    max_shapes = {}
    for entry in data:
        for key, tensor in entry.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("expected all entries to be tensors")
            shape = tuple(tensor.shape)
            if key not in max_shapes:
                max_shapes[key] = shape
            else:
                # Expand max_shapes[key] tuple elementwise to max dimension
                max_shapes[key] = tuple(
                    max(old_dim, new_dim)
                    for old_dim, new_dim in zip(max_shapes[key], shape)
                )
    return max_shapes


def aggregate(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    ''' finds out maximal dimensions for each key in list of tensor-dicts
        collates all tensor-dicts to single tensor-dicts with padded tensors '''
    max_shapes = maxShapes(data)
    batch_size = len(data)
    ds = {
        k: torch.empty(batch_size, *shape, dtype=data[0][k].dtype)
        for k, shape in max_shapes.items()
    }
    for i, item in enumerate(data):
        for k in ds.keys():
            shape = max_shapes[k]
            ds[k][i] = padTensor(item[k], shape)
    return ds


# -----------------------------------------------------------------------------
# generators

def generate(
    batch_size: int,
    seed: int,
    mcmc: bool = False,
    bs_load: int = 1,
) -> list[dict[str, torch.Tensor]]:
    """Generate a [batch_size] list of mixed effects datasets
    with varying n, m, d, q."""
    # init
    torch.manual_seed(seed)
    data = []
    iterator = tqdm(range(batch_size))
    if not mcmc:
        iterator.set_description(f"{part:02d}/{cfg.iterations:02d}")

    # presample hyperparams
    if cfg.varied:
        d = sampleInt(batch_size, 1, cfg.max_d)
        q = sampleIntBatched(0, d.clamp(max=cfg.max_q))
    else:
        d = sampleInt(batch_size, cfg.max_d, cfg.max_d)
        q = sampleInt(batch_size, cfg.max_q, cfg.max_q)
    m = sampleInt(batch_size // bs_load, cfg.min_m, cfg.max_m).repeat_interleave(
        bs_load
    )
    n = sampleInt((batch_size, cfg.max_m), cfg.min_n, cfg.max_n)
    d, m, n, q = d.tolist(), m.tolist(), n.tolist(), q.tolist()
    nu_ffx = D.Uniform(-20, 20).sample((batch_size, cfg.max_d))
    tau_ffx0 = D.Uniform(0.1, 30).sample((batch_size, 1))
    tau_ffx1 = D.Uniform(0.1, 20).sample((batch_size, cfg.max_d - 1))
    tau_ffx = torch.cat([tau_ffx0, tau_ffx1], dim=-1)
    tau_rfx = D.Uniform(0.1, 20).sample((batch_size, cfg.max_d))
    tau_eps = D.Uniform(0.1, 20).sample((batch_size,))
    if cfg.toy:  # smaller ranges
        nu_ffx = sampleInt((batch_size, cfg.max_d), 0, 0).float()
        tau_ffx0 = D.Uniform(0.1, 5).sample((batch_size, 1))
        tau_ffx1 = D.Uniform(0.1, 5).sample((batch_size, cfg.max_d - 1))
        tau_ffx = torch.cat([tau_ffx0, tau_ffx1], dim=-1)
        tau_rfx = D.Uniform(0.1, 1).sample((batch_size, cfg.max_d))
        tau_eps = D.Uniform(0.1, 1).sample((batch_size,))
        if cfg.mono:  # make each dim the same
            tau_ffx += tau_ffx.mean(-1, keepdim=True) - tau_ffx
            tau_rfx += tau_rfx.mean(-1, keepdim=True) - tau_rfx

    # sample datasets
    for i in iterator:
        ds = None
        attempts = 0
        okay = False
        d_i, q_i, m_i, n_i = d[i], q[i], m[i], n[i][: m[i]]
        while not okay:
            ds = MixedEffects(
                nu_ffx[i, :d_i],
                tau_ffx[i, :d_i],
                tau_eps[i],
                tau_rfx[i, :q_i],
                d_i, q_i, m_i, n_i,
                use_default=cfg.toy,
            ).sample()
            okay = ds["okay"]
            if not okay:
                attempts += 1
                if attempts > 20:
                    okay = True
                    print(f"\nWarning: outlier ds with sd(y)={ds['y'].std(0):.2f}")
        data += [ds]
    
    # optionally fit mcmc
    if mcmc:
        print('Starting MCMC sampling...')
        for i, ds in enumerate(tqdm(data)):
            mcmc_results = fitMCMC(ds, seed=i)
            ds.update(mcmc_results)
    return data


def generateSemi(
    ds_pre: dict[str, torch.Tensor],
    batch_size: int,
    seed: int,
    mcmc: bool = False,
    bs_load: int = 1,
) -> list[dict[str, torch.Tensor]]:
    """Generate a [batch_size] list of mixed effects datasets
    with varying n, m, d, q."""
    # init
    torch.manual_seed(seed)
    data = []
    features = ds_pre["X"]
    groups = ds_pre["groups"]
    this_ni = groups.unique(return_counts=True)[1]
    this_m = len(this_ni)
    this_n = max(this_ni)
    iterator = tqdm(range(batch_size))
    if not mcmc:
        iterator.set_description(f"{part:02d}/{cfg.iterations:02d}")

    # unpack
    if cfg.varied:
        d = sampleInt(batch_size, 1, cfg.max_d)
        q = sampleIntBatched(0, d.clamp(max=cfg.max_q))
    else:
        d = sampleInt(batch_size, cfg.max_d, cfg.max_d)
        q = sampleInt(batch_size, cfg.max_q, cfg.max_q)
    max_m = min(cfg.max_m, this_m)
    max_n = min(cfg.max_n, this_n)
    min_n = min(cfg.min_n, this_n)
    m = sampleInt(batch_size // bs_load, cfg.min_m, max_m).repeat_interleave(bs_load)
    n = sampleInt((batch_size, cfg.max_m), min_n, max_n)

    # presample priors
    nu_ffx = D.Uniform(-20, 20).sample((batch_size, cfg.max_d))
    tau_ffx0 = D.Uniform(0.1, 30).sample((batch_size, 1))
    tau_ffx1 = D.Uniform(0.1, 20).sample((batch_size, cfg.max_d - 1))
    tau_ffx = torch.cat([tau_ffx0, tau_ffx1], dim=-1)
    tau_rfx = D.Uniform(0.1, 10).sample((batch_size, cfg.max_d))
    tau_eps = D.Uniform(1e-3, 10).sample((batch_size,))

    d, m, n, q = d.tolist(), m.tolist(), n.tolist(), q.tolist()

    # use features
    use = sampleInt(batch_size, 0, 1).bool()

    # sample datasets
    for i in iterator:
        ds = None
        attempts = 0
        okay = False
        use_i = use[i] if part > 0 else True
        features_i = features if use_i else None
        groups_i = groups if use_i else None
        while not okay:
            d_i, q_i, m_i, n_i = d[i], q[i], m[i], n[i][: m[i]]
            ds = MixedEffects(
                nu_ffx[i, :d_i],
                tau_ffx[i, :d_i],
                tau_eps[i],
                tau_rfx[i, :q_i],
                d_i, q_i, m_i, n_i,
                features=features_i,
                groups=groups_i,
            ).sample()
            okay = ds["okay"]
            if not okay:
                attempts += 1
                if attempts > 20:
                    okay = True
                    print(f"\nWarning: outlier ds with sd(y)={ds['y'].std(0):.2f}")
        data += [ds]
        
    # optionally fit mcmc
    if mcmc:
        print('Starting MCMC sampling...')
        for i, ds in enumerate(tqdm(data)):
            mcmc_results = fitMCMC(ds, seed=i)
            ds.update(mcmc_results)
    return data


# =============================================================================

if __name__ == "__main__":
    os.makedirs(Path("..", "outputs", "data"), exist_ok=True)
    cfg = setup()
    assert cfg.type == "mfx", f"{cfg.type} not implemented"
    assert cfg.bs_train % cfg.bs_load == 0, (
        "storage batch size must be divisible by loading batch size"
    )
    part = cfg.begin
    raw = None
    if cfg.semi:
        assert cfg.d_tag, "must provide data tag for semi-synthetic data"
        path = Path("real", f"{cfg.d_tag}.npz")
        raw = RealDataset(path=path).raw()

    # generate test dataset
    if cfg.begin == -1:
        if cfg.semi and raw is not None:
            print("Generating semi-synthetic test set...")
            ds_test = generateSemi(raw, cfg.bs_test, part, mcmc=True)
            ds_test = aggregate(ds_test)
            fn = getFileName("test-semi", -1)
            np.savez_compressed(fn, **ds_test, allow_pickle=True)
            print(f"\nSaved semi-synthetic test set to {fn}")
        else:
            print("Generating test set...")
            ds_test = generate(cfg.bs_test, part, mcmc=True)
            ds_test = aggregate(ds_test)
            fn = getFileName("test", -1)
            np.savez_compressed(fn, **ds_test, allow_pickle=True)
            print(f"\nSaved test set to {fn}")
        exit()

    # generate validation dataset
    if cfg.begin == 0:
        if cfg.semi and raw is not None:
            print("Generating semi-synthetic validation set...")
            ds_val = generateSemi(raw, cfg.bs_val, part, mcmc=False)
            ds_val = aggregate(ds_val)
            fn = getFileName("val-semi", 0)
            np.savez_compressed(fn, **ds_val, allow_pickle=True)
            print(f"\nSaved semi-synthetic validation set to {fn}")
        else:
            print("Generating validation set...")
            ds_val = generate(cfg.bs_val, part, mcmc=False)
            ds_val = aggregate(ds_val)
            fn = getFileName("val", 0)
            np.savez_compressed(fn, **ds_val, allow_pickle=True)
            print(f"\nSaved validation set to {fn}")
        cfg.begin += 1

    # potentially stop after that
    if cfg.iterations == 0:
        exit()

    # generate training datasets
    if not cfg.semi:
        print(f"Generating {cfg.iterations} training partitions of {cfg.bs_train} datasets each...")
        for part in range(cfg.begin, cfg.iterations + 1):
            ds_train = generate(cfg.bs_train, part, bs_load=cfg.bs_load)
            ds_train = aggregate(ds_train)
            fn = getFileName("train", part)
            np.savez_compressed(fn, **ds_train, allow_pickle=True)
            print(f"Saved training set to {fn}")


