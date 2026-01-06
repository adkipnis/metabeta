import torch
from torch import nn
from metabeta.models.normalizingflows import Affine


class Coupling(nn.Module):
    ''' Single Coupling Step:
        1. Condition parameters on x1
        2. Parameterically transform x2
        3. return both with log determinant of Jacobian
    '''
    def __init__(
        self,
        split_dims: tuple[int, int],
        d_context: int = 0,
        net_kwargs: dict = {},
        transform: str = 'affine',
    ):
        super().__init__()
        if transform == 'affine':
            self.transform = Affine(split_dims, d_context, net_kwargs)

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor,
                 condition: torch.Tensor | None = None,
                 mask2: torch.Tensor | None = None,
                 inverse: bool = False):
        parameters = self.transform.propose(x1, condition)
        x2, log_det = self.transform(x2, parameters, mask2=mask2, inverse=inverse)
        return (x1, x2), log_det

    def forward(self, x1, x2, condition=None, mask2=None):
        return self(x1, x2, condition, mask2, inverse=False)

    def inverse(self, x1, x2, condition=None, mask2=None):
        return self(x1, x2, condition, mask2, inverse=True)


class DualCoupling(nn.Module):
    ''' Dual Coupling Step:
        1. Split inputs along pivot
        2. Apply first coupling step to split
        3. Apply second coupling step to swapped outputs
        4. return joint outputs and log determinants
    '''
    def __init__(
        self,
        d_data: int,
        d_context: int = 0,
        net_kwargs: dict = {},
        transform: str = 'affine',
    ):
        super().__init__()
        self.pivot = d_data // 2
        split_dims = (self.pivot, d_data - self.pivot)
        self.coupling1 = Coupling(
            split_dims=split_dims,
            d_context=d_context,
            net_kwargs=net_kwargs,
            transform=transform,
        )
        self.coupling2 = Coupling(
            split_dims=(split_dims[1], split_dims[0]),
            d_context=d_context,
            net_kwargs=net_kwargs,
            transform=transform,
        )

    def __call__(self, x: torch.Tensor,
