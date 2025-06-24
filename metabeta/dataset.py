from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def padTensor(tensor: torch.Tensor, shape: tuple, value=0) -> torch.Tensor:
    assert len(tensor.shape) == len(shape),\
        "Input tensor and target shape must have the same number of dimensions"
    pad_size = [max(s - t, 0) for s, t in zip(reversed(shape), reversed(tensor.shape))]
    padding = []
    for p in pad_size:
        padding.extend([0, p])  # Pad at the end of each dimension
    return F.pad(tensor, padding, value=value)


class LMDataset(Dataset):
    def __init__(self,
                 data: List[dict],
                 info: dict,
                 permute: bool = True,
                 ) -> None:
        self.info = info
        self.permute = permute
        self.n_max = self.info['max_n']
        self.d_max = self.info['max_d']
        self.m_max = self.info['max_m']
        self.q_max = self.info['max_q']
        self.len_max = self.n_max * self.m_max
        
        self.fx_type = 'mfx' if 'rfx' in data[0] else 'ffx'
        self.data = [self.preprocess(item) for item in data]

    def __len__(self) -> int:
       return len(self.data)
   
    def __getitem__(self, idx) -> dict:
        out = self.data[idx]
        return out

    def getPermutation(self, d: int):
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

        # pad inputs
        y = padTensor(item['y'], (length,)).unsqueeze(-1)
        X = padTensor(item['X'], (length, width))

        # pad targets
        ffx = padTensor(item['ffx'], (width,))
        sigma_error = item['sigma_error']

        # optinally permute
        if self.permute:
            idx = self.getPermutation(width)
            X = X[..., idx]
            ffx = ffx[idx]

        # masks
        mask_n = (y != 0.).squeeze()
        mask_d = (ffx != 0.).squeeze()

        # outputs
        out = {'n': item['n'], 'd': item['d'],
               'y': y, 'X': X,
               'ffx': ffx,
               'sigma_error': sigma_error,
               'mask_n': mask_n, 'mask_d': mask_d,
               }

        # optionally include analytical posterior
        if 'analytical' in item:
            _analytical = item['analytical']
            _ffx = _analytical['ffx']
            mu = padTensor(_ffx['mu'], (width,))
            Sigma = padTensor(_ffx['Sigma'], (width, width))
            if self.permute:
                mu = mu[idx] # type: ignore
                Sigma = Sigma[idx][..., idx] # type: ignore
            _ffx.update({'mu': mu, 'Sigma': Sigma})
            out.update({'analytical': _analytical})
        
        # optionally include mcmc posterior
        if 'mcmc' in item:
            _mcmc = item['mcmc']
            _ffx = padTensor(_mcmc['ffx'], (width, 4000))
            _sigma_error = _mcmc['sigma_error']
            if self.permute:
                _ffx = _ffx[idx]
            out['mcmc'] = {'global': torch.cat([_ffx, _sigma_error])}
        return out


    def preprocessMfx(self, item: dict) -> dict:
        length = self.len_max
        width = self.d_max + 1 # add bias term
        length_ = self.m_max

        # pad inputs entries
        y = padTensor(item['y'], (length,)).unsqueeze(-1)
        X = padTensor(item['X'], (length, width))
        Z = item['X'][..., :item['q']]
        Z = padTensor(Z, (length, width))
        groups = padTensor(item['groups'], (length,), value=-1)
        n_i = padTensor(item['n_i'], (length_,))

        # pad target entries
        ffx = padTensor(item['ffx'], (width,))
        rfx = padTensor(item['rfx'], (length_, width))
        sigmas_rfx = padTensor(item['sigmas_rfx'], (width,))
        sigma_error = item['sigma_error']

        # optinally permute
        if self.permute:
            idx = self.getPermutation(width)
            X = X[..., idx]
            ffx = ffx[idx]
            Z = Z[..., idx]
            rfx = rfx[..., idx]
            sigmas_rfx = sigmas_rfx[idx]

        # masks
        mask_n = (y != 0.).squeeze()
        mask_d = (ffx != 0.).squeeze()

        # outputs
        out = {'n': item['n'], 'm': item['m'], 'n_i': n_i,
               'd': item['d'], 'q': item['q'],
               'y': y, 'X': X, 'Z': Z, 'groups': groups,
               'ffx': ffx, 'rfx': rfx,
               'sigmas_rfx': sigmas_rfx, 'sigma_error': sigma_error,
               'mask_n': mask_n, 'mask_d': mask_d,
               }
        
        # optionally include mcmc posterior
        if 'mcmc' in item:
            _mcmc = item['mcmc']
            _ffx = padTensor(_mcmc['ffx'], (width, 4000))
            _sigma_error = _mcmc['sigma_error']
            _sigmas_rfx =  padTensor(_mcmc['sigmas_rfx'], (width, 4000))
            _rfx =  padTensor(_mcmc['rfx'], (width, length_, 4000))
            if self.permute:
                _ffx = _ffx[idx]
                _sigmas_rfx = _sigmas_rfx[idx]
                _rfx = _rfx[idx]
            out['mcmc'] = {
                'global': torch.cat([_ffx, _sigma_error, _sigmas_rfx]),
                'local': _rfx,
                }
        
        out = self.deepen(out)
        return out
    
    
    def deepen(self, item: dict):
        ''' split dimension for n (observations)
            into m (groups) and n_i (obs. per group)'''
        length = self.n_max
        width = self.d_max + 1
        m = item['m']
        masks = [item['groups'] == i for i in range(m)]
        
        # stacked arrays
        item['y'] = self._split(item['y'], masks, (length, 1))
        item['X'] = self._split(item['X'], masks, (length, width))
        Z = torch.clone(item['X'])
        Z[..., item['q']:] = 0.
        item['Z'] = Z
        
        # masks
        item['mask_n'] = (item['y'] != 0.).squeeze(-1)
        mask_m = torch.zeros((self.m_max,)).bool()
        mask_m[:m] = True
        item['mask_m'] = mask_m
        return item
        
        
    def _split(self, t: torch.Tensor, masks: List[torch.Tensor], shape: tuple):
        subsets = [padTensor(t[mask], shape).unsqueeze(0)
                   for mask in masks]
        pads = [torch.zeros(shape).unsqueeze(0)
                for _ in range(self.m_max - len(masks))]
        return torch.cat(subsets + pads, dim=0)
         
    
