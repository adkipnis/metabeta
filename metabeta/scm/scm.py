from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
from torch import nn
from metabeta.scm import CauseSampler, getPosthocLayers
from metabeta.simulation.utils import standardize, checkConstant

phl = getPosthocLayers()

class NoiseLayer(nn.Module):
    def __init__(self, sigma: float | torch.Tensor):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        noise = torch.randn_like(x) * self.sigma
        return x + noise
    
    
def sanityCheck(x: torch.Tensor) -> bool:
    okay = not checkConstant(x.detach().numpy()).any()
    return okay

@dataclass
class SCM(nn.Module):
    ''' simplified version of the MLP-based structural causal model
    in https://github.com/soda-inria/tabicl'''
    def __init__(self,
                 # data dims
                 n_samples: int,
                 n_features: int,

                 # causes
                 n_causes: int = 10, # number of units in initial layer
                 cause_dist: str = 'uniform', # [mixed, normal, uniform]
                 fixed: bool = False, # fixed moments of causes

                 # MLP architecture
                 n_layers: int = 8,
                 n_hidden: int = 32,
                 activation: Callable = nn.Tanh,

                 # weight initialization and feature extraction
                 sigma_w: float = 1.0, # for weight initialization
                 contiguous: bool = False, # sample adjacent features
                 blockwise: bool = True, # use blockwise dropout
                 p_dropout: float = 0.1, # dropout probability for weights

                 # Gaussian noise
                 sigma_e: float = 0.01, # for additive noise
                 vary_sigma_e: bool = True, # allow noise to vary per units
                 **kwargs):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_causes = n_causes
        self.cause_dist = cause_dist
        self.fixed = fixed
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = activation
        self.sigma_w = sigma_w
        self.contiguous = contiguous
        self.blockwise = blockwise
        self.p_dropout = p_dropout
        self.sigma_e = sigma_e
        self.vary_sigma_e = vary_sigma_e
 
        # make sure to have enough hidden units
        self.n_hidden = max(self.n_hidden, 2 * self.n_features)

        # init sampler for root nodes
        self.cs = CauseSampler(self.n_samples, self.n_causes,
                               dist=self.cause_dist,
                               fixed=self.fixed)

        # build layers
        layers = [self._buildLayer(self.n_causes)]
        for _ in range(self.n_layers - 1):
            layers += [self._buildLayer()]
        self.layers = nn.Sequential(*layers)

        # initialize weights
        with torch.no_grad():
            self._initLayers()


    def _buildLayer(self, input_dim: int = 0) -> nn.Module:
        # Affine() ->  AdditiveNoise() -> Activation()
        if input_dim == 0:
            input_dim = self.n_hidden
        affine_layer = nn.Linear(input_dim, self.n_hidden)
        sigma_e = self.sigma_e
        if self.vary_sigma_e:
            sigma_e = (torch.randn((self.n_hidden,)) * self.sigma_e).abs()
        noise_layer = NoiseLayer(sigma_e)
        return nn.Sequential(affine_layer, noise_layer, self.activation())


    def _initLayers(self):
        # init linear weights either with regular droput or blockwise dropout
        for i, (_, param) in enumerate(self.layers.named_parameters()):
            if param.dim() == 2: # skip biases
                if self.blockwise:
                    self._initLayerBlockDropout(param)
                else:
                    self._initLayer(param, i>0)


    def _initLayer(self, param, use_dropout: bool = True) -> None:
        p = self.p_dropout if use_dropout else 0.
        p = min(p, 0.99)
        sigma_w = self.sigma_w / ((1 - p) ** 0.5)
        nn.init.normal_(param, std=sigma_w)
        param *= torch.bernoulli(torch.full_like(param, 1 - p))


    def _initLayerBlockDropout(self, param) -> None:
        # blockwise weight dropout for higher dependency between features
        nn.init.zeros_(param)
        max_blocks = np.ceil(np.sqrt(min(param.shape)))
        n_blocks = np.random.randint(1, max_blocks)
        block_size = [dim // n_blocks for dim in param.shape]
        units_per_block = block_size[0] * block_size[1]
        keep_prob = (n_blocks * units_per_block) / param.numel()
        sigma_w = self.sigma_w / (keep_prob**0.5)
        for block in range(n_blocks):
            block_slice = tuple(slice(dim * block, dim * (block + 1))
                                for dim in block_size)
            nn.init.normal_(param[block_slice], std=sigma_w)

    def sample(self) -> torch.Tensor:
        while True:
            causes = self.cs.sample()  # (seq_len, num_causes)

            # pass through each mlp layer
            outputs = [causes]
            for layer in self.layers:
                h = layer(outputs[-1])
                h = torch.where(h.isnan() | h.abs().isinf(), 0, h)
                outputs.append(h)
            outputs = outputs[1:]  # remove causes

            # extract features
            outputs = torch.cat(outputs, dim=-1) # (n, units)
            n_units = outputs.shape[-1]
            if self.contiguous:
                start = np.random.randint(0, n_units - self.n_features)
                perm = start + torch.randperm(self.n_features)
            else:
                perm = torch.randperm(n_units-1)
            indices = perm[:self.n_features]
            x = outputs[:, indices]
            if sanityCheck(x):
                return x


class Posthoc(nn.Module):
    def __init__(self,
                 n_features: int,
                 p_posthoc: float = 0.2, # probability of posthoc transformation
                 **kwargs):
        super().__init__()

        # posthoc transformations
        self.n_features = n_features
        self.n_posthoc = np.random.binomial(n_features, p_posthoc)
        layers = []
        for _ in range(self.n_posthoc):
            cfg = {
                'n_in': n_features,
                'n_out': np.random.randint(1, 3),
                'standardize': True,
                # TODO levels, sigma
                }
            layer = np.random.choice(phl, replace=True)

            layers.append(layer(**cfg))
        self.transformations = nn.ModuleList(layers)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        if self.n_posthoc > 0:
            out = []
            for t in self.transformations:
                h = t(x)
                if sanityCheck(h):
                    out.append(h)
            z = torch.cat(out, dim=-1)
            x = torch.cat([x,z], dim=-1)
            idx = torch.randperm(x.shape[-1])[:self.n_features]
            x = x[..., idx]
        x = x.detach().numpy()
        x = standardize(x, axis=0)
        return x


# -----------------------------------------------
if __name__ == '__main__':
    from tqdm import tqdm
    from metabeta.scm import getActivations
    from metabeta.utils import logUniform, setSeed
    from metabeta.plot import plot
    setSeed(0)
    batches = 32

    # activation = RandomScaleFactory(GP)
    # activation = RandomChoiceFactory([GP] * 8)
    activations = getActivations()
    for _ in tqdm(range(batches)):
        config = {
            'n_samples': 512,
            'n_features': 5, #np.random.randint(3, 8),
            'n_causes': logUniform(2, 12, round=True),
            'cause_dist': np.random.choice(['uniform', 'normal', 'mixed']),
            'fixed': np.random.choice([True, False]),
            'n_hidden': logUniform(5, 30, round=True, add=4),
            'n_layers': np.random.randint(8, 32),
            'activation': np.random.choice(activations),
            'contiguous': np.random.choice([True, False]),
            'blockwise': np.random.choice([True, False]),
        }
        scm = SCM(**config)
        ph = Posthoc(**config)
        x = scm.sample()
        x = ph(x)

        # plot.correlation(x)
        plot.dataset(x, kde=False)
