from abc import abstractmethod
import torch
from torch import nn


class Transform(nn.Module):
    ''' Base template for transforms used in normalizing flows '''
    @abstractmethod
    def __call__(self, x: torch.Tensor,
                 condition: torch.Tensor | None = None,
                 mask: torch.Tensor | None = None,
                 inverse: bool = False,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ...

    def forward(self, x, condition=None, mask=None):
        return self(x, condition, mask, inverse=False)

    def inverse(self, x, condition=None, mask=None):
        return self(x, condition, mask, inverse=True)

    def _forwardMask(self, mask: torch.Tensor):
        return mask


