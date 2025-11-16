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


    def _step(
        self,
        x_synth: torch.Tensor,
        x_train: torch.Tensor,
    ) -> torch.Tensor:
        # detach gradient history
        x_synth = x_synth.detach().requires_grad_(True)

        # approximate energy using L2 distance
        distances = torch.cdist(x_synth, x_train)
        energy = distances.min(dim=1)[0] + distances.mean(1)
        energy_sum = energy.sum()

        # compute gradients
        grad = torch.autograd.grad(
            energy_sum,
            x_synth,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(x_synth)

        # update step
        noise = torch.randn_like(x_synth) * np.sqrt(2 * self.step_size)
        return x_synth - self.step_size * grad + self.noise_scale * noise


    def __call__(
