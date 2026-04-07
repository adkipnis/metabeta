from typing import Literal, Sequence
from dataclasses import dataclass
import torch
from metabeta.utils.dataloader import toDevice
from metabeta.utils.regularization import unconstrainedToCholeskyCorr


Proposed = dict[str, dict[str, torch.Tensor]]
Source = Literal['global', 'local']


def getMasks(
    data: dict[str, torch.Tensor], has_sigma_eps: bool = True
) -> dict[str, torch.Tensor | None]:
    out = {}
    out['ffx'] = data['mask_d']
    out['sigma_rfx'] = data['mask_q']
    out['rfx'] = data['mask_mq']
    if has_sigma_eps:
        out['sigma_eps'] = None
        out['sigmas'] = torch.nn.functional.pad(data['mask_q'], (0, 1), value=True)
    else:
        out['sigmas'] = data['mask_q']
    out['global'] = torch.cat([out['ffx'], out['sigmas']], dim=-1)
    return out


def getNames(source: str, num: int, has_sigma_eps: bool = True) -> list[str]:
    if source == 'ffx':
        return [rf'$\beta_{{{i}}}$' for i in range(num)]
    elif source == 'sigmas':
        names = [rf'$\sigma_{i}$' for i in range(num)]
        if has_sigma_eps:
            names.append(r'$\sigma_\epsilon$')
        return names
    elif source == 'rfx':
        return [rf'$\alpha_{{{i}}}$' for i in range(num)]
    else:
        raise ValueError(f'source {source} unknown.')


def getCorrRfxNames(q: int) -> list[str]:
    """Names for the lower-triangle correlation pairs: ρ_{01}, ρ_{02}, ρ_{12}, ..."""
    return [rf'$\rho_{{{j}{i}}}$' for i in range(q) for j in range(i)]


def getAllNames(d: int, q: int, has_sigma_eps: bool = True) -> dict[str, list[str]]:
    return {
        'ffx': getNames('ffx', d),
        'sigmas': getNames('sigmas', q, has_sigma_eps),
        'rfx': getNames('rfx', q),
    }


def joinSigmas(data: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [data['sigma_rfx']]
    if 'sigma_eps' in data:
        parts.append(data['sigma_eps'].unsqueeze(-1))
    return torch.cat(parts, dim=-1)


def joinGlobals(data: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [data['ffx'], data['sigma_rfx']]
    if 'sigma_eps' in data:
        parts.append(data['sigma_eps'].unsqueeze(-1))
    return torch.cat(parts, dim=-1)


class Proposal:
    def __init__(
        self,
        proposed: dict[str, dict[str, torch.Tensor]],
        has_sigma_eps: bool = True,
        d_corr: int = 0,
    ) -> None:
        self.data = proposed
        self.has_sigma_eps = has_sigma_eps
        self.d_corr = d_corr
        self.q = proposed['local']['samples'].shape[-1]
        D = proposed['global']['samples'].shape[-1]
        self.d = D - self.q - (1 if has_sigma_eps else 0) - d_corr
        self.is_results = {}
        self.tpd: float | None = None

    @property
    def n_samples(self) -> int:
        return self.samples_g.shape[-2]

    def __repr__(self) -> str:
        return f'ProposalPosterior(n_samples={self.n_samples})'

    def to(self, device: str | torch.device) -> None:
        self.data['global'] = toDevice(self.data['global'], device)
        self.data['local'] = toDevice(self.data['local'], device)

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
        return self.samples_g[..., self.d : self.d + self.q]

    @property
    def sigma_eps(self) -> torch.Tensor:
        if not self.has_sigma_eps:
            raise AttributeError('sigma_eps not available for this likelihood')
        return self.samples_g[..., self.d + self.q]

    @property
    def sigmas(self) -> torch.Tensor:
        end = self.d + self.q + (1 if self.has_sigma_eps else 0)
        return self.samples_g[..., self.d : end]

    @property
    def corr_rfx(self) -> torch.Tensor | None:
        """Correlation matrix reconstructed from unconstrained samples, shape (..., q, q).

        Returns None when the model was built without posterior_correlation.
        """
        if self.d_corr == 0:
            return None
        L = unconstrainedToCholeskyCorr(self.samples_g[..., -self.d_corr :], self.q)
        return L @ L.mT

    @property
    def rfx(self) -> torch.Tensor:
        return self.samples_l

    @property
    def parameters(self) -> tuple[torch.Tensor, ...]:
        if self.has_sigma_eps:
            return self.ffx, self.sigma_rfx, self.sigma_eps, self.rfx
        return self.ffx, self.sigma_rfx, self.rfx

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
        out['sigma_rfx'] = x[..., self.d : self.d + self.q]
        if self.has_sigma_eps:
            out['sigma_eps'] = x[..., self.d + self.q]
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
    has_sigma_eps = [p.has_sigma_eps for p in proposals]
    assert len(set(has_sigma_eps)) == 1, 'proposals disagree whether they have sigma_eps'
    d_corrs = [p.d_corr for p in proposals]
    assert len(set(d_corrs)) == 1, 'proposals disagree on d_corr'
    samples_g = torch.cat([p.samples_g for p in proposals], dim=-2)
    samples_l = torch.cat([p.samples_l for p in proposals], dim=-2)
    log_prob_g = torch.cat([p.log_prob_g for p in proposals], dim=-1)
    log_prob_l = torch.cat([p.log_prob_l for p in proposals], dim=-1)
    proposed = {
        'global': {'samples': samples_g, 'log_prob': log_prob_g},
        'local': {'samples': samples_l, 'log_prob': log_prob_l},
    }
    return Proposal(proposed, has_sigma_eps=has_sigma_eps[0], d_corr=d_corrs[0])


def concatProposalsBatch(proposals: list[Proposal]) -> Proposal:
    assert len(proposals) > 0, 'at least one proposal is required'
    has_sigma_eps = proposals[0].has_sigma_eps
    max_m = max(proposal.samples_l.shape[1] for proposal in proposals)

    samples_l_padded = []
    log_prob_l_padded = []
    for proposal in proposals:
        samples_l = proposal.samples_l
        log_prob_l = proposal.log_prob_l
        b, m, s, q = samples_l.shape
        if m < max_m:
            pad_samples = samples_l.new_zeros((b, max_m, s, q))
            pad_samples[:, :m] = samples_l
            samples_l = pad_samples

            pad_log_prob = log_prob_l.new_zeros((b, max_m, s))
            pad_log_prob[:, :m] = log_prob_l
            log_prob_l = pad_log_prob

        samples_l_padded.append(samples_l)
        log_prob_l_padded.append(log_prob_l)

    proposed = {
        'global': {
            'samples': torch.cat([proposal.samples_g for proposal in proposals], dim=0),
            'log_prob': torch.cat([proposal.log_prob_g for proposal in proposals], dim=0),
        },
        'local': {
            'samples': torch.cat(samples_l_padded, dim=0),
            'log_prob': torch.cat(log_prob_l_padded, dim=0),
        },
    }
    merged = Proposal(proposed, has_sigma_eps=has_sigma_eps, d_corr=proposals[0].d_corr)

    common_keys: set[str] | None = None
    for proposal in proposals:
        keys = set(proposal.is_results.keys())
        if common_keys is None:
            common_keys = keys
        else:
            common_keys &= keys

    if common_keys:
        for key in common_keys:
            values = [proposal.is_results[key] for proposal in proposals]
            if all(torch.is_tensor(value) for value in values):
                try:
                    merged.is_results[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    continue
    return merged


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
    prior_nll: torch.Tensor
    posterior_nll: torch.Tensor
    sample_efficiency: torch.Tensor | None
    pareto_k: torch.Tensor | None
    pp_fit: torch.Tensor | None = None  # R² (normal) | AUC (bernoulli) | deviance (poisson)
    tpd: float | None = None  # time per dataset

    def averageOverAlpha(
        self,
        nested: dict[float, dict[str, torch.Tensor]],
        absolute: bool = False,
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
    def ece(self) -> dict[str, torch.Tensor]:  # expected coverage error
        return self.averageOverAlpha(self.coverage_error)

    @property
    def lcr(self) -> dict[str, torch.Tensor]:  # log coverage ratio
        return self.averageOverAlpha(self.log_coverage_ratio)

    @property
    def abs_lcr(
        self,
    ) -> dict[str, torch.Tensor]:  # non-negative version of above for optimization
        return self.averageOverAlpha(self.log_coverage_ratio, absolute=True)

    @property
    def mnll(self) -> float:  # median posterior NLL
        return self.posterior_nll.median().item()

    @property
    def meff(self) -> None | float:  # median sample efficiency for IS
        if self.sample_efficiency is not None:
            return self.sample_efficiency.median().item()

    @property
    def mk(self) -> None | float:  # median pareto k for PSIS
        if self.pareto_k is not None:
            return self.pareto_k.median().item()

    @property
    def mfit(self) -> None | float:  # median R² or AUC
        if self.pp_fit is not None:
            return self.pp_fit.nanmedian().item()


def dictMean(td: dict[str, torch.Tensor]) -> float:
    values = list(td.values())
    for i, v in enumerate(values):
        if v.dim() == 0:
            values[i] = v.unsqueeze(0)
    return torch.cat(values).mean().item()
