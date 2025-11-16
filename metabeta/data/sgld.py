from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def checkDiscrete(x: np.ndarray, tol: float = 1e-12) -> torch.Tensor:
    diffs = np.abs(x - x.round())
    return (diffs < tol).all(axis=0)

