import math
import torch
from torch import distributions as D
from metabeta.utils.evaluation import Proposed, numFixed
from metabeta.utils.regularization import dampen


class ImportanceSampler:
    def __init__(
        self,
        data: dict[str, torch.Tensor],
        constrain: bool = True,
        eps: float = 1e-12,
    ) -> None:
        self.constrain = constrain
        self.eps = eps

        # prior
        self.nu_ffx  = data['nu_ffx'].unsqueeze(-2) # (b, 1, d)
        self.tau_ffx = data['tau_ffx'].unsqueeze(-2) + self.eps # (b, 1, d)
        self.tau_rfx = data['tau_rfx'].unsqueeze(-2) + self.eps # (b, 1, q)
        self.tau_eps = data['tau_eps'].unsqueeze(-1) + self.eps # (b, 1)

        # observations
        self.X = data['X'] # (b, m, n, d)
        self.Z = data['Z'] # (b, m, n, q)
        self.y = data['y'].unsqueeze(-1) # (b, m, n, 1)

        # masks
        self.mask_d = data['mask_d'].unsqueeze(-2) # (b, 1, d)
        self.mask_q = data['mask_q'].unsqueeze(-2) # (b, 1, q)
        mask_mq = data['mask_m'].unsqueeze(-1) & data['mask_q'].unsqueeze(-2)
        self.mask_mq = mask_mq.unsqueeze(-2) # (b, m, 1, q)
        self.mask_m = data['mask_m'].unsqueeze(-1) # (b, m, 1)
        self.mask_n = data['mask_n'].unsqueeze(-1) # (b, m, n, 1)

    def logPriorFfx(self, ffx: torch.Tensor) -> torch.Tensor:
        # ffx (b, s, d)
        dist = D.Normal(self.nu_ffx, self.tau_ffx)
        lp = dist.log_prob(ffx)
        lp = (lp * self.mask_d).sum(-1)
        return lp # (b, s)

    def logPriorSigmaRfx(self, sigma_rfx: torch.Tensor) -> torch.Tensor:
        # sigma_rfx (b, s, q)
        dist = D.HalfNormal(scale=self.tau_rfx)
        lp = dist.log_prob(sigma_rfx)
        lp = (lp * self.mask_q).sum(-1)
        return lp # (b, s)

    def logPriorSigmaEps(self, sigma_eps: torch.Tensor) -> torch.Tensor:
        # sigma_eps (b, s)
        dist = D.StudentT(df=4, loc=0, scale=self.tau_eps)
        lp = dist.log_prob(sigma_eps) + math.log(2.0)
        return lp # (b, s)

    def logPriorRfx(self, rfx: torch.Tensor, sigma_rfx: torch.Tensor) -> torch.Tensor:
        # rfx (b, m, s, q), sigma_rfx (b, s, q)
        dist = D.Normal(loc=0, scale=sigma_rfx.unsqueeze(1) + self.eps)
        lp = dist.log_prob(rfx)
        lp = (lp * self.mask_mq).sum(1)
        lp = lp.sum(-1)
        return lp # (b, s)

