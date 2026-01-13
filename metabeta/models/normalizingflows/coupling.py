import torch
from torch import nn
from metabeta.models.normalizingflows import (
    Transform, ActNorm, Permute, LU, Affine, RationalQuadratic, BaseDist
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
        elif transform == 'spline':
            self.transform = RationalQuadratic(split_dims, d_context, net_kwargs)
        else:
            raise NotImplementedError(
                'only affine and spline transforms are supported')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                 context: torch.Tensor | None = None,
                 mask2: torch.Tensor | None = None,
                 inverse: bool = False):
        x2, log_det = self.transform(
            x1, x2, context=context, mask2=mask2, inverse=inverse)
        return (x1, x2), log_det

    def inverse(self, x1, x2, context=None, mask2=None):
        return self(x1, x2, context, mask2, inverse=True)


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

    def _split(self, x: torch.Tensor):
        return x[..., :self.pivot], x[..., self.pivot:]

    def forward(self, x, condition=None, mask=None, inverse=False):
        x1, x2 = self._split(x)
        mask1, mask2 = None, None
        if mask is not None:
            mask1, mask2 = self._split(mask)
        if inverse:
            (x2, x1), log_det2 = self.coupling2.inverse(x2, x1, condition, mask1)
            (x1, x2), log_det1 = self.coupling1.inverse(x1, x2, condition, mask2)
        else:
            (x1, x2), log_det1 = self.coupling1.forward(x1, x2, condition, mask2)
            (x2, x1), log_det2 = self.coupling2.forward(x2, x1, condition, mask1)
        x = torch.cat([x1, x2], dim=-1)
        log_det = log_det1 + log_det2
        return x, log_det, mask


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
            if transform == 'spline':
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

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, condition=None, mask=None):
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        for flow in self.flows:
            x, ld, mask = flow.forward(x, condition=condition, mask=mask)
            log_det = log_det + ld
        return x, log_det, mask

    def inverse(self, x, condition=None, mask=None):
        log_det = torch.zeros(x.shape[:-1], device=x.device)
        for flow in reversed(self.flows):
            x, ld, mask = flow.inverse(x, condition=condition, mask=mask) # type: ignore
            log_det = log_det + ld
        return x, log_det, mask

    def _forwardMask(self, mask: torch.Tensor) -> torch.Tensor:
        for flow in self.flows:
            mask = flow._forwardMask(mask) # type: ignore
        return mask

    def logProb(
        self, z: torch.Tensor, log_det: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ''' joint log density of normalized target '''
        log_prob = self.base_dist.logProb(z)
        if mask is not None:
            log_prob *= mask
        return log_prob.sum(dim=-1) + log_det

    def loss(
            self,
            x: torch.Tensor,
            condition: torch.Tensor | None = None,
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ''' forward KL Loss (aka NLL) '''
        z, log_det, mask = self.forward(x, condition, mask)
        return -self.logProb(z, log_det, mask)

    def sample(
        self,
        n_samples: int,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # determine shape
        base_shape = (1,)
        if context is not None:
            base_shape = context.shape[:-1]
        elif mask is not None:
            base_shape = mask.shape[:-1]
        shape = (*base_shape, n_samples, self.d_target)

        # prepare context
        if context is not None:
            context = context.unsqueeze(-2).expand(*base_shape, n_samples, -1)

        # prepare mask
        if mask is not None:
            if mask.dim() < len(shape):
                mask = mask.unsqueeze(-2).expand(*shape)
            mask_z = self._forwardMask(mask)
        else:
            mask_z = None

        # sample from base and apply mask in base space
        z = self.base_dist.sample(shape).to(self.device)
        if mask_z is not None:
            z = z * mask_z

        # project z back to x space
        x, log_det, _ = self.inverse(z, context, mask_z)

        # get probability in x-space
        log_q = self.logProb(z, log_det, mask_z)
        return x, log_q

