from typing import Literal, Sequence
from dataclasses import dataclass
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
    out['sigmas'] = torch.nn.functional.pad(data['mask_q'], (0, 1), value=True)
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
        
def getAllNames(d: int, q: int) -> dict[str, list[str]]:
    return {
        'ffx': getNames('ffx', d),
        'sigmas': getNames('sigmas', q),
        'rfx': getNames('rfx', q),
    }


def joinSigmas(data: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([data['sigma_rfx'], data['sigma_eps'].unsqueeze(-1)], dim=-1)


class Proposal:
    def __init__(self, proposed: dict[str, dict[str, torch.Tensor]]) -> None:
        self.data = proposed
        self.d = numFixed(proposed)
        self.q = proposed['local']['samples'].shape[-1]
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
    def pareto_k(self) -> torch.Tensor | None:
        return self.is_results.get('pareto_k')

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


def weightedQuantile(
    x: torch.Tensor,
    w: torch.Tensor,
    q: float | Sequence[float] | torch.Tensor = 0.5,
) -> torch.Tensor:
    if not isinstance(q, torch.Tensor):
        q_t = torch.tensor(q, dtype=x.dtype, device=x.device)
    else:
        q_t = q
    w = w.unsqueeze(-1)
    if x.dim() == 4:
        w = w.unsqueeze(1)
    # sort samples and weights
    x_sorted, order = torch.sort(x, dim=-2)
    w_sorted = torch.gather(w.expand_as(x), dim=-2, index=order)

    # search sorted
    x_last = x_sorted.movedim(-2, -1)
    cdf = torch.cumsum(w_sorted, dim=-2).movedim(-2, -1).contiguous()
    t = q_t * torch.ones_like(cdf[..., 0:1])
    idx = torch.searchsorted(cdf, t, right=False).clamp(max=x_last.shape[-1] - 1)
    out = torch.gather(x_last, dim=-1, index=idx).squeeze(-1)
    return out


@dataclass()
class EvaluationSummary:
    estimates: dict[str, torch.Tensor]
    nrmse: dict[str, torch.Tensor]
    corr: dict[str, torch.Tensor]
    credible_intervals: dict[float, dict[str, torch.Tensor]]
    coverage: dict[float, dict[str, torch.Tensor]]
    coverage_error: dict[float, dict[str, torch.Tensor]]
    log_coverage_ratio: dict[float, dict[str, torch.Tensor]]
    predictive_nll: torch.Tensor
    sample_efficiency: torch.Tensor | None
    pareto_k: torch.Tensor | None
    tpd: float | None = None   # time per dataset

    def averageOverAlpha(
        self, nested: dict[float, dict[str, torch.Tensor]], absolute: bool = False,
    ) -> dict[str, torch.Tensor]:
        out = {}
        alphas = list(nested.keys())
        params = list(nested[alphas[0]].keys())
        for param in params:
            param_values = [nested[alpha][param].unsqueeze(0) for alpha in alphas]
            cat = torch.cat(param_values)
            if absolute:
                cat = torch.abs(cat)
            out[param] = cat.mean(0)
        return out

    @property
    def ece(self) -> dict[str, torch.Tensor]:   # expected coverage error
        return self.averageOverAlpha(self.coverage_error)

    @property
    def lcr(self) -> dict[str, torch.Tensor]:   # log coverage ratio
        return self.averageOverAlpha(self.log_coverage_ratio)
    
    def abs_lcr(self) -> dict[str, torch.Tensor]:   # non-negative version of above for optimization
        return self.averageOverAlpha(self.log_coverage_ratio, absolute=True)

    @property
    def mnll(self) -> float:   # median posterior NLL
        return self.predictive_nll.median().item()

    @property
    def meff(self) -> None | float:   # median sample efficiency for IS
        if self.sample_efficiency is not None:
            return self.sample_efficiency.median().item()

    @property
    def mk(self) -> None | float:   # median pareto k for PSIS
        if self.pareto_k is not None:
            return self.pareto_k.median().item()



def dictMean(td: dict[str, torch.Tensor]) -> float:
    values = list(td.values())
    for i, v in enumerate(values):
        if v.dim() == 0:
            values[i] = v.unsqueeze(0)
    return torch.cat(values).mean().item()
