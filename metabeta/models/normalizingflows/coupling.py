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

