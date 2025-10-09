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
    def __init__(
        self, path: Path, max_d: int = 0, max_q: int = 0, permute: bool = True
    ):
        # load and store as tensors
        assert path.exists(), (
            f"File {path} does not exist, you must generate it first using generate.py"
        )
        data = np.load(path)
        data = {key: torch.from_numpy(data[key]) for key in data.files}

        # save properties
        self.len = data["y"].shape[0]
        self.max_n = int(data["n"].max())
        self.max_d = int(max(data["d"].max(), torch.tensor(max_d)))
        self.max_r = int((self.max_d - 1) * (self.max_d - 2) / 2)

        # precompute permutations
        self.permute = permute
        if permute:
            perm = [getPermutation(self.max_d) for _ in range(self.len)]
            unperm = [inversePermutation(p) for p in perm]
            data["perm"] = torch.stack(perm)
            data["unperm"] = torch.stack(unperm)
        else:
            data["perm"] = torch.arange(0, self.max_d).unsqueeze(0).expand(self.len, -1)
            data["unperm"] = data["perm"]

        # mfx extras
        self.is_mfx = "rfx" in data
        if self.is_mfx:
            self.max_n_i = int(data["n_i"].max())
            self.max_m = int(data["m"].max())
            self.max_q = int(max(data["q"].max(), torch.tensor(max_q)))
            data["rfx"] = padTensor(data["rfx"], (self.len, self.max_m, self.max_d))
            data["sigmas_rfx"] = padTensor(data["sigmas_rfx"], (self.len, self.max_d))
            data["tau_rfx"] = padTensor(data["tau_rfx"], (self.len, self.max_d))

        # preprocess for faster access
        self.data = [self.preprocess(data, i) for i in range(len(self))]

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        return self.data[i]

    def preprocess(
        self, data: dict[str, torch.Tensor], i: int
    ) -> dict[str, torch.Tensor]:
        # sizes
        d = data["d"][i]
        q = data["q"][i]
        n = data["n"][i]
        m = data["m"][i]
        n_i = data["n_i"][i, :m]

        # inputs
        y = data["y"][i, :n]
        X = data["X"][i, :n]
        Z = X.clone()
        Z[:, q:] = 0
        non_empty = torch.ones(self.max_d, dtype=torch.bool)
        non_empty[self.max_q :] = False

        # parameters
        ffx = data["ffx"][i]
        sigma_eps = data["sigma_eps"][i]
        sigmas_rfx = data["sigmas_rfx"][i]
        rfx = data["rfx"][i, :m]
        cov_sum = data["cov_sum"][i]

        # priors
        nu_ffx = data["nu_ffx"][i]
        tau_ffx = data["tau_ffx"][i]
        tau_eps = data["tau_eps"][i]
        tau_rfx = data["tau_rfx"][i]

        # optinally permute
        perm = data["perm"][i]
        unperm = data["unperm"][i]
        if self.permute:
            X = X[..., perm]
            Z = Z[..., perm]
            ffx = ffx[perm]
            rfx = rfx[..., perm]
            sigmas_rfx = sigmas_rfx[perm]
            nu_ffx = nu_ffx[perm]
            tau_ffx = tau_ffx[perm]
            tau_rfx = tau_rfx[perm]

        # correlation
        R = torch.zeros(self.max_r)
        if d > 2:
            r = torch.corrcoef(X[:, 1:].permute(1, 0))
            row_idx, col_idx = torch.triu_indices(
                self.max_d - 1, self.max_d - 1, offset=1
            )
            R = r[row_idx, col_idx].nan_to_num()

        # remove empty dims for rfx-related tensors
        rfx_mask = perm[non_empty] if self.permute else non_empty
        rfx = rfx[:, rfx_mask]
        sigmas_rfx = sigmas_rfx[rfx_mask]
        tau_rfx = tau_rfx[rfx_mask]

        # masks
        mask_m = torch.zeros(self.max_m, dtype=torch.bool)
        mask_m[:m] = True
        mask_d = (ffx != 0.0).squeeze()
        mask_q = torch.ones(self.max_d, dtype=torch.bool)
        mask_q[q:] = False
        mask_q = mask_q[rfx_mask]
        mask_c = data["categorial"][i]

        # outputs
        out = {
            # sizes
            'm': m, 'n': n, 'n_i': n_i, 'd': d, 'q': q,  
            # inputs
            # 'y_long': y, 'X_long': X, 'Z_long': Z,
            'y': y, 'X': X, 'Z': Z, 'R': R,
            # global params
            'ffx': ffx, 'sigmas_rfx': sigmas_rfx, 'sigma_eps': sigma_eps, 
            # local params
            'rfx': rfx, 'cov_sum': cov_sum,
            # priors
            'nu_ffx': nu_ffx, 'tau_ffx': tau_ffx,
            'tau_eps': tau_eps, 'tau_rfx': tau_rfx, 
            # masks
            'mask_m': mask_m, 'mask_d': mask_d, 'mask_q': mask_q, 'mask_c': mask_c,
            # permutations
            'perm': perm, 'unperm': unperm,
            # snr
            'rnv': data['rnv'][i],
            }

        # optionally include mcmc posterior
        if "mcmc_ffx" in data:
            m_ffx = data["mcmc_ffx"][i]
            m_sigma_eps = data["mcmc_sigma_eps"][i]
            m_sigmas_rfx = data["mcmc_sigmas_rfx"][i]
            m_rfx = data["mcmc_rfx"][i].permute(1, 0, 2)

            # take the chain with the fewest outliers
            m_tensors = [m_ffx, m_sigma_eps, m_sigmas_rfx, m_rfx]
            m_tensors = findBestChain(m_tensors)
            m_ffx, m_sigma_eps, m_sigmas_rfx, m_rfx = m_tensors

            if self.permute:
                mc_samples = m_ffx.shape[-1]
                m_ffx = m_ffx[perm]
                m_sigmas_rfx = padTensor(m_sigmas_rfx, (self.max_d, mc_samples))
                m_sigmas_rfx = m_sigmas_rfx[perm][rfx_mask]
                m_rfx = padTensor(m_rfx, (self.max_m, self.max_d, mc_samples))
                m_rfx = m_rfx[perm][:, rfx_mask]
            out["mcmc_global"] = torch.cat([m_ffx, m_sigmas_rfx, m_sigma_eps])
            out["mcmc_local"] = m_rfx

        # unfold to 4d and
        out = self.unfold(out)
        out["Z"] = out["Z"][..., rfx_mask]
        return out

    def unfold(self, item: dict[str, torch.Tensor]):
        """split dimension for n (observations)
        into m (groups) and n_i (obs. per group)"""
        m = int(item["m"])
        n = self.max_n_i
        d = self.max_d
        counts = item["n_i"]

        # stacked arrays
        y = split(item["y"], counts, shape=[m, n])
        X = split(item["X"], counts, shape=[m, n, d])
        Z = split(item["Z"], counts, shape=[m, n, d])
        mask_n = y != 0.0
        item.update({"y": y, "X": X, "Z": Z, "mask_n": mask_n})
        return item


# =============================================================================
if __name__ == "__main__":
    import time
    from metabeta.utils import dsFilename

    def measure(fn, args=(), kwargs={}):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        end = time.perf_counter()
        print(f"{end - start:.2f}s")
        return out

    # mixed effects example
    fn_t = dsFilename('mfx', 'train', b=32, m=30, n=70, d=3, q=1,
                      size=int(4e3), part=1, tag='r', outside=True)
    ds_t = measure(LMDataset, (fn_t,))
    item = ds_t[0]
    dl_t = DataLoader(ds_t, batch_size=32, shuffle=False)
    batch = next(iter(dl_t))

    # check dl test
    fn_t = dsFilename('mfx', 'test', b=1, m=30, n=70, d=3, q=1,
                    size=128, part=-1, tag='r', outside=True)
    ds_t = LMDataset(fn_t, permute=True)
    dl_t = DataLoader(ds_t, batch_size=256, shuffle=False,
                      collate_fn=getCollater(autopad=True))
    batch = next(iter(dl_t))

    # check semi-synthetic
    fn_s = dsFilename('mfx', 'val-semi', b=1, m=30, n=70, d=5, q=2, size=500,
                      part=0, outside=True)
    ds_s = LMDataset(fn_s, permute=True)
    dl_s = DataLoader(ds_s, batch_size=500, shuffle=False)
    batch = next(iter(ds_s))

