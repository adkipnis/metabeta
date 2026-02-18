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
    out['rfx'] = data['mask_mq']
    return out


def getNames(source: str, num: int) -> list[str]:
    if source == 'ffx':
        return [rf'$\beta_{{{i}}}$' for i in range(num)]
    elif source == 'sigmas':
        return [rf'$\sigma_{i}$' for i in range(num)] + [r'$\sigma_\epsilon$']
    elif source == 'rfx':
        return [rf'$\alpha_{{{i}}}$' for i in range(num)]
    else:
        raise ValueError(f'source {source} unknown.')


def joinSigmas(data: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([data['sigma_rfx'], data['sigma_eps'].unsqueeze(-1)], dim=-1)


class Proposal:
    def __init__(self, proposed: dict[str, dict[str, torch.Tensor]]) -> None:
        self.data = proposed
        self.d = numFixed(proposed)
        self.is_results = {}

    @property
    def n_samples(self) -> int:
        return self.sigma_eps.shape[-1]

    def __repr__(self) -> str:
        return f'ProposalPosterior(n_samples={self.n_samples})'

    @property
    def samples_g(self) -> torch.Tensor:
        return self.data['global']['samples']

    @property
    def samples_l(self) -> torch.Tensor:
        return self.data['local']['samples']

    @property
    def log_prob_g(self) -> torch.Tensor:
        return self.data['global']['log_prob']

    @property
    def log_prob_l(self) -> torch.Tensor:
        return self.data['local']['log_prob']

    @property
    def ffx(self) -> torch.Tensor:
        return self.samples_g[..., : self.d]

    @property
    def sigma_rfx(self) -> torch.Tensor:
        return self.samples_g[..., self.d : -1]

    @property
    def sigma_eps(self) -> torch.Tensor:
        return self.samples_g[..., -1]

    @property
    def sigmas(self) -> torch.Tensor:
        return self.samples_g[..., self.d :]

    @property
    def rfx(self) -> torch.Tensor:
        return self.samples_l

    @property
    def parameters(self) -> tuple[torch.Tensor, ...]:
        return self.ffx, self.sigma_rfx, self.sigma_eps, self.rfx

    @property
    def log_probs(self) -> tuple[torch.Tensor, ...]:
        return self.log_prob_g, self.log_prob_l

    @property
    def weights(self) -> torch.Tensor | None:
        return self.is_results.get('weights', None)

    @property
    def efficiency(self) -> torch.Tensor | None:
        return self.is_results.get('sample_efficiency')

    @property
    def mean_efficiency(self) -> float | None:
        s_eff = self.efficiency
        if s_eff is not None:
            return s_eff.mean().item()

    def partition(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        out['ffx'] = x[..., : self.d]
        out['sigma_rfx'] = x[..., self.d : -1]
        out['sigma_eps'] = x[..., -1]
        return out

    def rescale(self, scale: torch.Tensor) -> None:
        for source in ('global', 'local'):
            samples = self.data[source]['samples']
            scale_ = scale.view(-1, *([1] * (samples.ndim - 1)))
            self.data[source]['samples'] *= scale_

    def subset(self, idx: torch.Tensor) -> None:
        b, s = idx.shape
        for source in ('global', 'local'):
            # samples
            samples = self.data[source]['samples']
            m, d = samples.shape[1], samples.shape[-1]
            index = idx.unsqueeze(-1).expand(b, s, d)
            if source == 'local':
                index = index.unsqueeze(1).expand(b, m, s, d)
            re_samples = torch.gather(samples, dim=-2, index=index)
            self.data[source]['samples'] = re_samples

            # log probs
            log_prob = self.data[source]['log_prob']
            index = idx.clone()
            if source == 'local':
                index = index.unsqueeze(1).expand(b, m, s)
            re_log_prob = torch.gather(log_prob, dim=-1, index=index)
            self.data[source]['log_prob'] = re_log_prob


def joinProposals(proposals: list[Proposal]) -> Proposal:
    samples_g = torch.cat([p.samples_g for p in proposals], dim=-2)
    samples_l = torch.cat([p.samples_l for p in proposals], dim=-2)
    log_prob_g = torch.cat([p.log_prob_g for p in proposals], dim=-1)
    log_prob_l = torch.cat([p.log_prob_l for p in proposals], dim=-1)
    proposed = {
        'global': {'samples': samples_g, 'log_prob': log_prob_g},
        'local': {'samples': samples_l, 'log_prob': log_prob_l},
    }
    return Proposal(proposed)

