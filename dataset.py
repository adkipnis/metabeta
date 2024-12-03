from typing import List, Tuple
import torch
from torch.utils.data import Dataset


def padTensor(tensor: torch.Tensor, shape: Tuple[int, int], pad_val: int = 0) -> torch.Tensor:
    ''' Pad a tensor with a constant value. '''
    padded = torch.full(shape, pad_val, dtype=torch.float32)
    padded[:tensor.shape[0], :tensor.shape[1]] = tensor
    return padded


class LMDataset(Dataset):
    def __init__(self, data: List[dict], max_samples: int, max_predictors: int) -> None:
        self.data = data
        self.max_samples = max_samples
        self.max_predictors = max_predictors
        self.seeds = torch.tensor([item['seed'] for item in data])

    def __len__(self) -> int:
       return len(self.data)

    def preprocess(self, item: dict) -> dict:
        predictors = item['predictors']
        y = item['y']
        params = item['params']
        return {
            'predictors': padTensor(predictors, (self.max_samples, self.max_predictors + 1)),
            'y': padTensor(y, (self.max_samples, 1)),
            'params': padTensor(params, (self.max_predictors + 1, 1)),
            'd': predictors.shape[1],
        }

    def __getitem__(self, idx) -> dict:
        data = self.data[idx]
        return self.preprocess(data)

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



