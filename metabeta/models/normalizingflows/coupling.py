import torch
from torch import nn
from metabeta.models.normalizingflows import (
    Transform, ActNorm, Permute, LU, Affine, BaseDist
)


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
        else:
            raise NotImplementedError()

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


class DualCoupling(Transform):
    ''' Dual Coupling Step:
        1. Split inputs along pivot
        2. Apply first coupling step to split
        3. Apply second coupling step to swapped outputs
        4. return joint outputs and log determinants
    '''
    def __init__(
        self,
        d_target: int,
        d_context: int = 0,
        net_kwargs: dict = {},
        transform: str = 'affine',
    ):
        super().__init__()
        self.pivot = d_target // 2
        split_dims = (self.pivot, d_target - self.pivot)
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

    def __call__(self, x, condition=None, mask=None, inverse=False):
        z = torch.empty_like(x)
        x1, x2 = x[..., :self.pivot], x[..., self.pivot:]
        mask1, mask2 = None, None
        if mask is not None:
            mask1, mask2 = mask[..., :self.pivot], mask[..., self.pivot:]
        if inverse:
            (x2, x1), log_det2 = self.coupling2.inverse(x2, x1, condition, mask1)
            (x1, x2), log_det1 = self.coupling1.inverse(x1, x2, condition, mask2)
        else:
            (x1, x2), log_det1 = self.coupling1.forward(x1, x2, condition, mask2)
            (x2, x1), log_det2 = self.coupling2.forward(x2, x1, condition, mask1)
        z[..., :self.pivot], z[..., self.pivot:] = x1, x2
        log_det = log_det1 + log_det2
        return z, log_det, mask


class CouplingFlow(nn.Module):
    ''' Normalizing Flow based on Coupling:
        ActNorm -> (LU) -> Permute -> DualCoupling 
        - forward: Conditionally map X to Z in base distribution
        - inverse: Conditionally map Z to X in target distribution
        - logProb: cheaply evaluate the density of X using the chain rule
        - sample: cheaply sample from X by sampling from Z + inverse pass
        '''
    def __init__(
        self,
        d_target: int,
        d_context: int = 0,
        n_blocks: int = 6,
        use_actnorm: bool = True,
        use_permute: bool = True,
        use_lu: bool = False,
        transform: str = 'affine', # type of coupling transform
        family: str = 'student', # family of base distribution
        trainable: bool = True, # train parameters of base distribution
        net_kwargs: dict = {},
    ):
        super().__init__()
        self.d_target = d_target
        self.base_dist = BaseDist(d_target, family=family, trainable=trainable)
        flows = []
        for _ in range(n_blocks):
            if use_actnorm:
                flows += [ActNorm(d_target)]
            if use_lu:
                flows += [LU(d_target, identity_init=True)]
            if use_permute:
                flows += [Permute(d_target)]
            flows += [
                DualCoupling(
                    d_target=d_target,
                    d_context=d_context,
                    transform=transform,
                    net_kwargs=net_kwargs,
                )
            ]
        self.flows = nn.ModuleList(flows)



