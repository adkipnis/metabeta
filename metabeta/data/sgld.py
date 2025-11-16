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
        self,
        X_train: np.ndarray | torch.Tensor,
        n_samples: int = 0,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.numpy()
            to_torch = True
        else:
            to_torch = False
            
        # prepare reference set
        X_scaled = self.scaler.fit_transform(X_train)
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)

        # init synthetic set
        x_std = np.std(X_scaled, axis=0)
        noise = np.random.normal(0, x_std * 0.1, X_scaled.shape)
        x_synth = X_scaled + noise
        x_synth = torch.tensor(x_synth, device=self.device, dtype=torch.float32)

        # optionally subsample
        if n_samples:
            indices = np.random.permutation(len(x_synth))[:n_samples]
            x_synth = x_synth[indices]

        # SGLD iterations with adaptive step size
        adaptive_step_size = self.step_size
        iterator = range(self.n_steps)
        if self.use_tqdm:
            iterator = tqdm(iterator)
        for step in iterator:
            x_synth = self._step(x_synth, x_train)
            if step % 100 == 0:
                adaptive_step_size *= 0.9
            
        # unstandardize
        X_synth = x_synth.detach().cpu().numpy()
        X_synth = self.scaler.inverse_transform(X_synth)
        
        # round previously discrete columns
        discrete = checkDiscrete(X_train)
        X_synth[:, discrete] = X_synth[:, discrete].round()
        
        # optionally back to torch
        if to_torch:
            X_synth = torch.tensor(X_synth)
        return X_synth

