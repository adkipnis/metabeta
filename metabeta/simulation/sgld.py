from dataclasses import dataclass, field
import numpy as np
import torch
from metabeta.utils.preprocessing import Standardizer, checkContinuous

@dataclass
class SGLD:
    ''' Stochastig gradient Langevin dynamics generator for X
        simplified version of https://github.com/sebhaan/TabPFGen '''
    n_steps: int = 500
    step_size: float = 0.01
    noise_scale: float = 0.01
    standardizer: Standardizer = field(
        default_factory=lambda: Standardizer(axis=0))
    device: str = 'cpu'

    def __call__(
        self,
        X_train: np.ndarray, # (n, d)
        rng: np.random.Generator,
        n_samples: int = 0,
    ) -> np.ndarray:
        assert n_samples <= len(X_train), 'n_samples must be smaller than n'

        # prepare reference set
        X_scaled = self.standardizer.forward(X_train)
        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)

        # init synthetic set
        x_std = np.std(X_scaled, axis=0)
        noise = rng.normal(0, x_std * 0.1, X_scaled.shape)
        x_synth = X_scaled + noise
        x_synth = torch.tensor(x_synth, device=self.device, dtype=torch.float32)

        # optionally subsample
        if n_samples:
            indices = rng.permutation(len(x_synth))[:n_samples]
            x_synth = x_synth[indices]

        # SGLD iterations with adaptive step size
        self.adaptive_step_size = float(self.step_size)
        iterator = range(self.n_steps)
        for step in iterator:
            x_synth = self._step(x_synth, x_train)
            if step % 100 == 0:
                self.adaptive_step_size *= 0.9

        # unstandardize
        X_synth = x_synth.detach().cpu().numpy()
        X_synth = self.standardizer.backward(X_synth)
 
        # round previously discrete columns
        discrete = ~checkContinuous(X_train)
        X_synth[:, discrete] = X_synth[:, discrete].round()

        return X_synth


    def _step(self, x_synth: torch.Tensor, x_train: torch.Tensor) -> torch.Tensor:
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
        eps = self.adaptive_step_size
        noise = torch.randn_like(x_synth) * np.sqrt(2.0 * eps)
        x_synth = x_synth - eps * grad + self.noise_scale * noise
        return x_synth


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    seed = 1
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    sgld = SGLD()

    n = 200
    d = 3
    x = np.random.normal(size=(n,d))
    x_sim = sgld(x, rng=rng)

