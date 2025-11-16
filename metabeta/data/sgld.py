from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def checkDiscrete(x: np.ndarray, tol: float = 1e-12) -> torch.Tensor:
    diffs = np.abs(x - x.round())
    return (diffs < tol).all(axis=0)

class SGLD:
    ''' Stochastig gradient Langevin dynamics generator for X
        simplified version of https://github.com/sebhaan/TabPFGen '''
    n_steps: int = 250
    step_size: float = 0.01
    noise_scale: float = 0.01
    device: str = 'cpu'
    use_tqdm: bool = False
    scaler = StandardScaler()


