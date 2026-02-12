from typing import Literal
import torch


Proposed = dict[str, dict[str, torch.Tensor]]
Source = Literal['global', 'local']


def numFixed(proposed: Proposed) -> int:
    q = proposed['local']['samples'].shape[-1]
    D = proposed['global']['samples'].shape[-1]
    d = D - q - 1
    return int(d)


def getMasks(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
    out = {}
    out['ffx'] = data['mask_d']
    out['sigma_rfx'] = data['mask_q']
    out['sigma_eps'] = None
    out['rfx'] = data['mask_m'].unsqueeze(-1) & data['mask_q'].unsqueeze(-2)
    return out


class Proposal:
    def __init__(self, proposed: dict[str, dict[str, torch.Tensor]]) -> None:
        self.data = proposed
        self.d = numFixed(proposed)
        self.s = self.sigma_eps().shape[-1]

    def __repr__(self) -> str:
        return f'ProposalPosterior(n_samples={self.s})'

    def samples(self, source: Source) -> torch.Tensor:
        return self.data[source]['samples']

    def log_probs(self, source: Source) -> torch.Tensor:
        return self.data[source]['log_prob']

    def ffx(self) -> torch.Tensor:
        return self.samples('global')[..., :self.d]

    def sigma_rfx(self) -> torch.Tensor:
        return self.samples('global')[..., self.d:-1]

    def sigma_eps(self) -> torch.Tensor:
        return self.samples('global')[..., -1]

    def sigmas(self) -> torch.Tensor:
        return self.samples('global')[..., self.d:]

    def partition(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        out['ffx'] = x[..., :self.d]
        out['sigma_rfx'] = x[..., self.d:-1]
        out['sigma_eps'] = x[..., -1]
        return out

    def rfx(self) -> torch.Tensor:
        return self.samples('local')

    def rescale(self, scale: torch.Tensor) -> None:
        for source in ('global', 'local'):
            samples = self.data[source]['samples']
            scale_ = scale.view(-1, *([1] * (samples.ndim - 1)))
            self.data[source]['samples'] *= scale_
#         # # update log_probs
#         # k = data['mask_q'].sum(-1)
#         # if source == 'global':
#         #     k += data['mask_d'].sum(-1) + 1
#         #     log_det = (data['sd_y'].log() * k).unsqueeze(-1)
#         # else:
#         #     log_det = (data['sd_y'].log() * k).unsqueeze(-1).unsqueeze(-1)
#         # proposed[source]['log_prob'] -= log_det
