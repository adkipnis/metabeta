from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import dInput


def padTensor(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    assert len(tensor.shape) == len(shape),\
        "Input tensor and target shape must have the same number of dimensions"
    pad_size = [max(s - t, 0) for s, t in zip(reversed(shape), reversed(tensor.shape))]
    padding = []
    for p in pad_size:
        padding.extend([0, p])  # Pad at the end of each dimension
    return F.pad(tensor, padding)


class LMDataset(Dataset):
    def __init__(self,
                 data: List[dict],
                 info: dict,
                 permute: bool = False,
                 ) -> None:
        self.info = info
        self.permute = permute
        self.n_max = self.info['max_n']
        self.d_max = self.info['max_d']
        self.fx_type = 'mfx' if 'rfx' in data[0] else 'ffx'
        self.data = [self.preprocess(item) for item in data]

    def __len__(self) -> int:
       return len(self.data)
   
    def getPermutation(self, d: int):
        # idx = torch.randperm(d)
        idx = torch.randperm(d-1) + 1
        zero = torch.zeros((1,), dtype=idx.dtype)
        idx = torch.cat([zero, idx])
        return idx
   
    def preprocess(self, item: dict) -> dict:
       if self.fx_type == 'ffx':
           return self.preprocessFfx(item)
       elif self.fx_type == 'mfx':
           return self.preprocessMfx(item)
       else:
           raise ValueError(f'Unsupported fx type {self.fx_type} for preprocessing')
    
    def preprocessFfx(self, item: dict) -> dict:
        length = self.n_max
        width = self.d_max + 1 # add bias term

        X = padTensor(item['X'], (length, width))
        y = padTensor(item['y'], (length,)).unsqueeze(-1)
        ffx = padTensor(item['ffx'], (width,))
        sigma_error_emp = padTensor(item['sigma_error_emp'], (length,)).unsqueeze(-1)
        mask = (y == 0.).squeeze()

        if self.permute:
            idx = self.getPermutation(X.shape[-1])
            X = X[..., idx]
            ffx = ffx[..., idx]

        out = {'n': item['n'], 'd': item['d'],
               'X': X, 'y': y, 'ffx': ffx,
               'sigma_error': item['sigma_error'],
               'sigma_error_emp': sigma_error_emp,
               'mask': mask,
               }
        
        # optionally include analytical posterior
        if 'optimal' in item:
            optimal = item['optimal']
            ffx = optimal['ffx']
            mu = padTensor(ffx['mu'], (width,))
            Sigma = padTensor(ffx['Sigma'], (width, width))
            ffx.update({'mu': mu, 'Sigma': Sigma})
            out.update({'optimal': optimal})
        return out

    def deepen(self, item: dict, depth: int, length: int, width: int):
        masks = [item['groups'] == i for i in range(item['m'])]
        y_subsets = [padTensor(item['y'][mask], (length,)).unsqueeze(0)
                   for mask in masks]
        y = padTensor(torch.cat(y_subsets), (depth, length)).unsqueeze(-1)
        X_subsets = [padTensor(item['X'][mask], (length, width)).unsqueeze(0)
                   for mask in masks]
        X = padTensor(torch.cat(X_subsets), (depth, length, width))
        Z = torch.clone(X)
        Z[..., item['q']:] = 0.
        masks = padTensor(torch.cat([m.unsqueeze(0) for m in masks]), (depth, length))
        return y, X, Z, masks

    def preprocessMfx(self, item: dict) -> dict:
        if self.permute:
            raise NotImplementedError()
    
        length = self.n_max
        width = self.d_max + 1 # add bias term
        
        # pad inputs entries
        y = padTensor(item['y'], (length,)).unsqueeze(-1)
        X = padTensor(item['X'], (length, width))
        Z = torch.clone(X)
        Z[..., item['q']:] = 0.
        mask = (y == 0.).squeeze()
        
        # pad flat entries
        ffx = padTensor(item['ffx'], (width,))
        rfx = padTensor(item['rfx'], (length, width))
        groups = padTensor(item['groups'], (length,))
        S = padTensor(item['S'], (width,))
        S_emp = padTensor(item['S_emp'], (length, width))
        sigma_error_emp = padTensor(item['sigma_error_emp'], (length,)).unsqueeze(-1)
            
        out = {'n': item['n'], 'd': item['d'],
               'm': item['m'], 'q': item['q'],
               'y': y, 'X': X, 'Z': Z,
               'ffx': ffx, 'rfx': rfx,
               'groups': groups, 'mask': mask,
               'S': S, 'S_emp': S_emp,
               'sigma_error': item['sigma_error'],
               'sigma_error_emp': sigma_error_emp,
               }
        return out
    
    # def preprocessMfx(self, item: dict) -> dict:
    #     if self.permute:
    #         raise NotImplementedError()

    #     length = self.n_max
    #     width = self.d_max + 1 # add bias term
    #     depth = self.n_max // 10
        
    #     # get padded subsets
    #     y, X, Z, mask = self.deepen(item, depth, length, width)

    #     # pad flat entries
    #     ffx = padTensor(item['ffx'], (width,))
    #     rfx = padTensor(item['rfx'], (length, width))
    #     groups = padTensor(item['groups'], (length,))
    #     S = padTensor(item['S'], (width,))
    #     S_emp = padTensor(item['S_emp'], (length, width))
    #     sigma_error_emp = padTensor(item['sigma_error_emp'], (length,)).unsqueeze(-1)
            
    #     out = {'n': item['n'], 'd': item['d'],
    #            'm': item['m'], 'q': item['q'],
    #            'y': y, 'X': X, 'Z': Z,
    #            'ffx': ffx, 'rfx': rfx,
    #            'groups': groups, 'mask': mask,
    #            'S': S, 'S_emp': S_emp,
    #            'sigma_error': item['sigma_error'],
    #            'sigma_error_emp': sigma_error_emp,
    #            }
    #     return out
    
    def __getitem__(self, idx) -> dict:
        out = self.data[idx]
        return out

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
    from utils import dsFilenameVal
    filename = dsFilenameVal("mfx", 8, 50, 0)
    ds_raw = torch.load(filename, weights_only=False)
    ds = LMDataset(**ds_raw)


