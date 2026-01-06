from abc import abstractmethod
import torch
from torch import nn
from metabeta.models.feedforward import FlowMLP, FlowResidualNet


class CouplingTransform(nn.Module):
    ''' Base class for coupling transforms:
        - build conditioner network
        - propose transform parameters
        - directionally apply transform parameters to inputs
    '''
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self.split_dims = split_dims
        self.d_context = d_context

        # setup conditioner
        assert net_kwargs['net_type'] in ['mlp', 'residual']
        self.buildConditioner(net_kwargs.copy())

    @abstractmethod
    def buildConditioner(self, net_kwargs: dict) -> None:
        ...

    @abstractmethod
    def propose(
        self, x1: torch.Tensor, condition: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        ...

    @abstractmethod
    def __call__(
        self,
        x2: torch.Tensor,
        params: tuple[torch.Tensor, ...],
        mask2: torch.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...

    def forward(self, x2, params, mask2):
        return self(x2, params, mask2, inverse=False)

    def inverse(self, z2, params, mask2):
        return self(z2, params, mask2, inverse=True)


class Affine(CouplingTransform):
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
        alpha: float = 1.0, # softclamping denominator
    ):
        super().__init__(split_dims, d_context, net_kwargs)
        self.alpha = alpha

