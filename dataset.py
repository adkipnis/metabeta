from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def padTensor(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    assert len(tensor.shape) == len(shape),\
        "Input tensor and target shape must have the same number of dimensions"
    pad_size = [max(s - t, 0) for s, t in zip(reversed(shape), reversed(tensor.shape))]
    padding = []
    for p in pad_size:
        padding.extend([0, p])  # Pad at the end of each dimension
    return F.pad(tensor, padding)


class LMDataset(Dataset):
    def __init__(self, data: List[dict], max_samples: int, max_predictors: int, permute: bool = False) -> None:
        self.data = data
        self.max_samples = max_samples
        self.max_predictors = max_predictors
        self.seeds = torch.tensor([item['seed'] for item in data])
        self.permute = permute
        self.data_p = [self.preprocess(item) for item in data]

    def __len__(self) -> int:
       return len(self.data)

    def preprocess(self, item: dict) -> dict:
        n = self.max_samples
        d = item['X'].shape[1]
        d_max = self.max_predictors + 1

        X = padTensor(item['X'], (n, d_max))
        y = padTensor(item['y'], (n,))
        beta = padTensor(item['beta'], (d_max,))

        if self.permute:
            seed = item['seed']
            torch.manual_seed(seed)
            indices = torch.randperm(d_max)
            X = X[:, indices]
            beta = beta[indices]

        out = {'d': d, 'X': X, 'y': y, 'beta': beta, 'sigma_error': item['sigma_error']}

        # optionally include random effects
        if "rfx" in item:
            q = item['rfx'].shape[1]
            rfx = padTensor(item['rfx'], (n, d_max))
            S = padTensor(item['S'], (d_max,))
            # unique = int(d_max * (d_max + 1) / 2)
            # S = padTensor(item['S'], (unique,))
            out.update({'q': q, 'rfx': rfx, 'S': S})
            if self.permute:
                raise ValueError('Permutation not implemented for rfx datasets')

        # optionally include analytical posterior
        if 'mu_n' in item:
            mu_n = padTensor(item['mu_n'], (n, d_max))
            Sigma_n = padTensor(item['Sigma_n'], (n, d_max, d_max))
            a_n = item['a_n']
            b_n = item['b_n']
            if self.permute:
                mu_n = mu_n[:, indices] # type: ignore
                Sigma_n = Sigma_n[:, indices] # type: ignore
                Sigma_n = Sigma_n[:, :, indices] # type: ignore
            out.update({'mu_n': mu_n, 'Sigma_n': Sigma_n, 'a_n': a_n, 'b_n': b_n})

        return out

    def __getitem__(self, idx) -> dict:
        return self.data_p[idx]

    def randomSplit(self, split: float = 0.9, shuffle: bool = False) -> Tuple['LMDataset', 'LMDataset']:
        ''' Split the dataset into two parts, randomly. '''
        num_samples = len(self)
        split_idx = int(num_samples * split)
        if shuffle:
            indices = torch.randperm(num_samples)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            train_data = [self.data[i] for i in train_indices]
            test_data = [self.data[i] for i in test_indices]
        else:
            train_data = self.data[:split_idx]
            test_data = self.data[split_idx:]
        train_dataset = LMDataset(train_data, self.max_samples, self.max_predictors)
        test_dataset = LMDataset(test_data, self.max_samples, self.max_predictors)
        return train_dataset, test_dataset


if __name__ == '__main__':
    from pathlib import Path
    fname = Path('data', 'dataset-val-noise=variable.pt')
    ds_raw = torch.load(fname, weights_only=False)
    ds = LMDataset(**ds_raw)
