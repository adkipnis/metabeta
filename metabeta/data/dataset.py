from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from metabeta.utils import padTensor, getPermutation, inversePermutation


def split(
    long: torch.Tensor,
    counts: torch.Tensor,
    shape: list[int],
    max_num: int | None = None,
) -> torch.Tensor:
    """splits the observations dimension of long into
    groups and padded observations per group"""
    max_num = shape[1]
    if shape[-1] != long.shape[-1]:
        new_shape = (*long.shape[:-1], shape[-1])
        long = padTensor(long, new_shape)
    out = torch.zeros(shape, dtype=long.dtype)
    idx = counts.unsqueeze(1) > torch.arange(max_num).unsqueeze(0)
    j = counts.sum()
    out[idx] = long[:j]
    return out


def autoPad(batch: list[dict[str, torch.Tensor]], key: str, device: str | torch.device = "cpu"):
    """automatically pad a tensor with zeros depending on its dim"""
    get_shape = None
    out = []
    dim = batch[0][key].dim()
    assert dim in range(4), f'unexpected dim: {dim}'
    shapes = [item[key].shape for item in batch]
    if dim == 0:
        out = [item[key] for item in batch]
    elif dim == 1:
        L = max(s[0] for s in shapes)
        get_shape = lambda item: (0, L - item[key].shape[0])
    elif dim == 2:
        N = max(s[0] for s in shapes)
        D = max(s[1] for s in shapes)
        get_shape = lambda item: (0, D - item[key].shape[1],
                                  0, N - item[key].shape[0])
    elif dim == 3:
        M = max(s[0] for s in shapes)
        N = max(s[1] for s in shapes)
        D = max(s[2] for s in shapes)
        get_shape = lambda item: (0, D - item[key].shape[2],
                                  0, N - item[key].shape[1],
                                  0, M - item[key].shape[0])
    if get_shape is not None:
        out = [pad(item[key], get_shape(item), value=0) for item in batch]
    return torch.stack(out).to(device)


def getCollater(autopad: bool = False, device: str | torch.device = "cpu"):
    if autopad:

        def collate_fn(batch: list[dict[str, torch.Tensor]]):  # type: ignore
            keys = batch[0].keys()
            return {k: autoPad(batch, k, device) for k in keys}
    elif device != "cpu":

        def collate_fn(batch: list[dict[str, torch.Tensor]]):
            batch_dict = default_collate(batch)
            return {k: v.to(device) for k, v in batch_dict.items()}
    else:
        collate_fn = default_collate  # type: ignore
    return collate_fn


def getDataLoader(
    filename: Path,
    batch_size: int,
    permute: bool = True,
    max_d: int = 0,
    max_q: int = 0,
    autopad: bool = False,
    device: str | torch.device = "cpu",
) -> DataLoader:
    collate_fn = getCollater(autopad, device)
    ds = LMDataset(filename, max_d=max_d, max_q=max_q, permute=permute)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def madOutliers(tensor: torch.Tensor, threshold: float = 3.0) -> int:
    """get number of median absolute distande (MAD) outliers"""
    median = tensor.median(-1)[0].unsqueeze(-1)
    abs_dev = (tensor - median).abs()
    mad = abs_dev.median(-1)[0].unsqueeze(-1)
    modified_z_scores = 0.6745 * abs_dev / mad
    out = (modified_z_scores > threshold).sum()
    return int(out)


def fewestOutliers(mc_samples: torch.Tensor, n_chains: int = 4) -> torch.Tensor:
    """get index of markov chain with fewest outliers"""
    tensors = mc_samples.chunk(n_chains, dim=-1)
    n_samples = tensors[0].shape[-1]
    num_outliers = torch.tensor([madOutliers(st.squeeze()) for st in tensors])
    no_variance = torch.tensor([st.squeeze().var(-1).sum() < 1e-6 for st in tensors])
    num_outliers[no_variance] = n_samples
    fewest_outliers_idx = num_outliers.min(0)[1]
    return fewest_outliers_idx


# def findBestChain(tensor_list: list[torch.Tensor], n_chains: int = 4) -> list[torch.Tensor]:
#     indices = torch.stack([fewestOutliers(t, n_chains=n_chains)
#                            for t in tensor_list])
#     idx_best = indices.mode()[0]
#     new_tensors = [t.chunk(n_chains, dim=-1)[idx_best] for t in tensor_list]
#     return new_tensors


def findBestChain(
    tensor_list: list[torch.Tensor], n_chains: int = 4
) -> list[torch.Tensor]:
    indices = torch.stack([fewestOutliers(t, n_chains=n_chains) for t in tensor_list])
    new_tensors = [t.chunk(n_chains, dim=-1)[i] for i, t in zip(indices, tensor_list)]
    return new_tensors


class LMDataset(Dataset):
