from copy import deepcopy
import numpy as np
import torch
from torch import nn
from metabeta.utils import (
    maskedMean,
    maskedStd,
    batchCovary,
    maskedSoftplus,
    maskedInverseSoftplus,
    dampen,
    nParams,
)
from metabeta.models.transformers import (
    BaseSetTransformer,
    SetTransformer,
    DualTransformer,
)
from metabeta.models.posteriors import Posterior, CouplingPosterior
from metabeta import plot

mse = nn.MSELoss()


# -----------------------------------------------------------------------------


# base approximator
class Approximator(nn.Module):
    posterior: Posterior | None = None
    stats: dict = {}

    def __init__(
        self,
        constrain: bool = True,
        use_standardization: bool = True,
    ):
        super().__init__()
        self.constrain = constrain
        self.use_standardization = use_standardization

    @staticmethod
    def modelID(s_dict: dict, p_dict: dict, m_dict: dict) -> str:
        prefix = ''
        suffix = ''
        if m_dict['tag']:
            suffix = '-' + m_dict['tag']
        summary = f'{prefix}{s_dict['type']}-{s_dict['n_blocks']}-{s_dict['d_model']}-{s_dict['d_ff']}*{s_dict['depth']}-{s_dict['d_output']}-{s_dict['n_heads']}-{s_dict['activation']}-{s_dict['dropout']}'
        posterior = f'{p_dict['type']}-{p_dict['n_blocks']}-{p_dict['d_ff']}*{p_dict['depth']}-{p_dict['activation']}-{p_dict['dropout']}'
        return f'mfx-d={m_dict['d']}-q={m_dict['q']}-{summary}-+-{posterior}-seed={m_dict['seed']}{suffix}'

    @property
    def device(self):
        return next(self.parameters()).device

    def inputs(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        '''prepare input tensor for the summary network'''
        raise NotImplementedError

    def targets(self, data: dict[str, torch.Tensor]):
        '''prepare target tensor for the posterior network'''
        raise NotImplementedError

    def forward(self, data: dict[str, torch.Tensor], sample: bool = False):
        raise NotImplementedError

    def standardize(self,
                    x: torch.Tensor,
                    name: str,
                    mask: torch.Tensor | None = None,
                    ) -> torch.Tensor:
        '''z-standardization specific for each dataset'''
        dim = tuple(range(1, x.dim() - 1))
        mean = maskedMean(x, dim, mask=mask)
        std = maskedStd(x, dim, mask=mask, mean=mean) + 1e-12
            
        # exclude categorical columns for standardization
        categorical = ((x == 0) | (x == 1)).all([1,2], keepdim=True)
        mean[categorical] = 0
        std[categorical] = 1
        
        # save moments
        self.stats[name] = {'mean': mean, 'std': std}
        
        # final step
        out = (x - mean) / std
        if mask is not None:
            out *= mask
        return out

    def unpackMoment(
        self, names_list: list[str], moment: str, device: str | None = None
    ) -> dict[str, torch.Tensor]:
        if device is None:
            return {name: self.stats[name][moment] for name in names_list}
        return {name: self.stats[name][moment].to(device) for name in names_list}

    def unpackMean(self, names_list, device=None):
        return self.unpackMoment(names_list, 'mean', device)

    def unpackStd(self, names_list, device=None):
        return self.unpackMoment(names_list, 'std', device)

    def moments(
        self,
        proposed: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''wrapper for location and scale of the posterior'''
        assert 'samples' in proposed, 'no samples in proposed'
        assert self.posterior is not None
        return self.posterior.getLocScale(proposed)

    def quantiles(
        self,
        proposed: dict[str, torch.Tensor],
        roots: list = [0.025, 0.975],
        calibrate: bool = False,
    ) -> torch.Tensor | None:
        '''wrapper for desired quantiles of the posterior'''
        if 'samples' in proposed:
            assert self.posterior is not None
            samples = proposed['samples'].clone()
            quantiles = self.posterior.getQuantiles(samples, roots, calibrate)
            return quantiles

    def ranks(
        self, proposed: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        assert self.posterior is not None
        return self.posterior.getRanks(proposed, targets)

    def plotRecovery(
        self,
        targets: torch.Tensor,
        names: list[str],
        means: torch.Tensor,
        color: str = 'darkgreen',
        alpha: float = 0.3,
        return_stats: bool = True,
    ) -> None | tuple[float, float]:
        return plot.recovery(targets, names, means, color, alpha, return_stats)

    def recoveryGrouped(
        self,
        targets: list[torch.Tensor],
        names: list[list[str]],
        means: list[torch.Tensor],
        titles: list[str] = [],
        marker: str = 'o',
        alpha: float = 0.2,
    ) -> None | tuple[float, float]:
        return plot.recoveryGrouped(targets, names, means, titles, marker, alpha)


class ApproximatorMFX(Approximator):
    def __init__(
        self,
        summarizer_g: BaseSetTransformer,  # global summarizer
        summarizer_l: BaseSetTransformer,  # local summarizer
        posterior_g: Posterior,  # global posterior
        posterior_l: Posterior,  # local posterior
        model_id: str,
        constrain: bool = True,  # constrains sigmas
        use_standardization: bool = True,  # standardizes inputs
    ):
        super().__init__(constrain, use_standardization)
        self.summarizer_g = summarizer_g
        self.summarizer_l = summarizer_l
        self.posterior_g = posterior_g
        self.posterior_l = posterior_l
        num_sum_g = nParams(self.summarizer_g)
        num_sum_l = nParams(self.summarizer_l)
        num_inf_g = nParams(self.posterior_g)
        num_inf_l = nParams(self.posterior_l)
        self.num_sum = num_sum_g + num_sum_l
        self.num_inf = num_inf_g + num_inf_l
        self.id = model_id

    @classmethod
    def build(cls, cfg: dict):
        s_dict = cfg['summarizer']
        p_dict = cfg['posterior']
        m_dict = cfg['general']
        d_ffx = m_dict['d']
        d_rfx = m_dict['q']
        model_id = cls.modelID(s_dict, p_dict, m_dict)
        cls.d = d_ffx
        cls.q = d_rfx
        cls.r = int((d_ffx - 1) * (d_ffx - 2) / 2)

        # 1. summary networks
        sum_type = s_dict['type']
        s_dict_l = s_dict.copy()
        d_input_l = 1 + (d_ffx - 1) + (d_rfx - 1)
        d_input_g = s_dict_l['d_output'] + 1  # num obs per group

        if sum_type == 'set-transformer':
            Summarizer = SetTransformer
        elif sum_type == 'dual-transformer':
            Summarizer = DualTransformer
        else:
            raise ValueError(f'unknown summary type {sum_type}')
        summarizer_l = Summarizer(d_input=d_input_l, **s_dict_l)
        summarizer_g = Summarizer(d_input=d_input_g, **s_dict)

        # dimension variables
        post_type = p_dict['type']
        d_var = 1 + d_rfx  # variance components
        prior_dims = (
            2 * d_ffx + d_var
        )  # ffx prior (nu, tau_f), rfx variance prior (tau_r), noise prior (tau_e)
        d_context_g = (
            s_dict['d_output'] + 2 + prior_dims
        )  # global conditional: global summary, num groups, num obs, priors
        d_context_l = (
            d_input_g + d_ffx + d_var
        )  # local conditional: local summary, global parameters

        # 2. posterior networks
        assert post_type in ['affine', 'spline'], 'unkown posterior type'
        posterior_g = CouplingPosterior(
            d_target=d_ffx + d_var,
            d_context=d_context_g,
            n_flows=p_dict['n_blocks'],
            transform=post_type,
            base_type=p_dict['base'],
            net_kwargs=p_dict,
        )
        p_dict_l = deepcopy(p_dict)
        # p_dict_l['flows'] //= 2
        # p_dict_l['d_ff'] //= 2
        # p_dict_l['depth'] = int(1.5 * p_dict['depth'])
        posterior_l = CouplingPosterior(
            d_target=d_rfx,
            d_context=d_context_l,
            n_flows=p_dict_l['n_blocks'],
            transform=post_type,
            base_type=p_dict_l['base'],
            net_kwargs=p_dict_l,
        )

        return cls(
            summarizer_g,
            summarizer_l,
            posterior_g,
            posterior_l,
            model_id=model_id,
            constrain=m_dict['constrain'],
            use_standardization=m_dict['standardize'],
        )

    @property
    def posterior(self):  # type: ignore
        return self.posterior_g

    @property
    def calibrator(self):
        return self.posterior_g.calibrator

    @property
    def calibrator_l(self):
        return self.posterior_l.calibrator

    def moments(
        self, proposed: dict[str, torch.Tensor], local: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''wrapper for location and scale of the posterior'''
        assert 'samples' in proposed, 'no samples in proposed'
        if local:
            return self.posterior_l.getLocScale(proposed)
        return self.posterior_g.getLocScale(proposed)

    def quantiles(
        self,
        proposed: dict[str, torch.Tensor],
        roots: list = [0.025, 0.975],
        calibrate: bool = False,
        local: bool = False,
        use_weights: bool = True,
    ) -> torch.Tensor | None:
        '''wrapper for desired quantiles of the posterior'''
        if 'samples' in proposed:
            samples = proposed['samples'].clone()
            weights = proposed.get('weights') if use_weights else None
            if local:
                p = self.posterior_l
                m = samples.shape[1]
                quantiles = [
                    p.getQuantiles(samples[:, i], roots, calibrate).unsqueeze(1)
                    for i in range(m)
                ]
                return torch.cat(quantiles, dim=1)
            p = self.posterior_g
            return p.getQuantiles(samples, roots, calibrate, weights=weights)

    def inputs(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        y = data['y'].unsqueeze(-1)
        X = data['X'][..., 1 : self.d]
        Z = data['Z'][..., 1 : self.q]
        if self.use_standardization:
            mask = data['mask_n'].unsqueeze(-1)
            y = self.standardize(y, 'y', mask)
            X = self.standardize(X, 'X', mask)
            Z = self.standardize(Z, 'Z', mask)
        out = torch.cat([y, X, Z], dim=-1)
        return out

    def names(self, data: dict[str, torch.Tensor], local: bool = False) -> np.ndarray:
        if local:
            names = [rf'$\alpha_{{{i}}}$' for i in range(data['rfx'].shape[2])]
        else:
            names = (
                [rf'$\beta_{{{i}}}$' for i in range(data['ffx'].shape[1])]
                + [rf'$\sigma_{i}$' for i in range(data['sigmas_rfx'].shape[1])]
                + [r'$\sigma_e$']
            )
        return np.array(names)

    def targets(
        self, data: dict[str, torch.Tensor], local: bool = False
    ) -> torch.Tensor:
        if local:
            out = data['rfx']
        else:
            out = [data['ffx'], data['sigmas_rfx'], data['sigma_eps'].unsqueeze(-1)]
            out = torch.cat(out, dim=-1)
        return out

    def addMetadata(
        self, summary: torch.Tensor, data: dict, local: bool = False
    ) -> torch.Tensor:
        '''append summary tensor with n_obs and priors'''
        if local:
            # number of group observations
            out = [summary]
            out += [data['n_i'].unsqueeze(-1).sqrt() / 10]

        else:
            # number of groups and total number of observations
            out = [summary]
            out += [
                data['m'].unsqueeze(-1).sqrt() / 10,
                data['n'].unsqueeze(-1).sqrt() / 10,
            ]

            # prior params
            nu_f = data['nu_ffx'].clone()
            tau_f = data['tau_ffx'].clone()
            tau_r = data['tau_rfx'].clone()
            tau_e = data['tau_eps'].unsqueeze(-1).clone()

            if self.use_standardization:
                # standardize priors
                b, d, q = len(nu_f), self.d, self.q
                std_y, std_X, std_Z = self.unpackStd(['y', 'X', 'Z']).values()
                nu_f /= std_y.view(b, 1)
                nu_f[:, 1:] *= std_X.view(b, d - 1)
                tau_f /= std_y.view(b, 1)
                tau_f[:, 1:] *= std_X.view(b, d - 1)
                tau_r /= std_y.view(b, 1)
                tau_r[:, 1:] *= std_Z.view(b, q - 1)
                tau_e /= std_y.view(b, 1)

                # add stds
                # out += [dampen(s.view(b,-1)) for s in [std_y, std_Z]]

            # reduce abolute size for better NN handling
            nu_f = dampen(nu_f)
            tau_f = dampen(tau_f)
            tau_r = dampen(tau_r)
            tau_e = dampen(tau_e)
            out += [nu_f, tau_f, tau_r, tau_e]
        return torch.cat(out, dim=-1)

    def preprocess(
        self, targets: torch.Tensor, data: dict[str, torch.Tensor], local: bool = False
    ) -> torch.Tensor:
        '''analytically standardize targets and constrain variance components'''
        targets = targets.clone()

        # prepare moments
        if self.use_standardization:
            mean_y, mean_X, mean_Z = self.unpackMean(['y', 'X', 'Z']).values()
            std_y, std_X, std_Z = self.unpackStd(['y', 'X', 'Z']).values()

        # local parameters
        if local:
            rfx = targets
            if self.use_standardization:
                b, q = len(rfx), self.q

                # standardize rfx
                rfx_ = rfx / std_y.view(b, 1, 1)
                rfx_[..., 1:] *= std_Z.view(b, 1, q - 1)
                mean_Zb = (mean_Z.view(b, 1, q - 1) * rfx[..., 1:]).sum(2)
                rfx_[..., 0] = (rfx[..., 0] + mean_Zb) / std_y.view(b, 1)

                # patch targets
                rfx = rfx_
            # put everything back together
            targets = rfx

        # global parameters
        else:
            b, d, q = len(targets), self.d, self.q
            ffx, sigmas_rfx, sigma_eps = (
                targets[:, :d],
                targets[:, d:-1],
                targets[:, -1:],
            )

            if self.use_standardization:
                # standardize ffx
                ffx_ = ffx / std_y.view(b, 1)
                ffx_[:, 1:] *= std_X.view(b, d - 1)
                mean_Xb = (mean_X.view(b, d - 1) * ffx[:, 1:]).sum(1)
                ffx_[:, 0] = (ffx[:, 0] + mean_Xb - mean_y.view(b)) / std_y.view(b)  # this might be too small with std_y

                # standardize sigmas
                sigma_eps /= std_y.view(b, 1)
                sigmas_rfx_ = sigmas_rfx / std_y.view(b, 1)
                sigmas_rfx_[:, 1:] *= std_Z.view(b, q - 1)  # random slopes

                # sigma intercept with covsum
                cov_sum = data['cov_sum']  # sum of the mean covariance between Z and rfx
                sigmas_rfx_[:, 0] = (
                    sigmas_rfx[:, 0].square() + cov_sum
                ).sqrt() / std_y.view(b)

                # patch targets
                ffx = ffx_
                sigmas_rfx = sigmas_rfx_

            # project positives to reals
            if self.constrain:
                sigmas_rfx = maskedInverseSoftplus(sigmas_rfx + 1e-6)
                sigma_eps = maskedInverseSoftplus(sigma_eps + 1e-6)

            # put everything back together
            targets = torch.cat([ffx, sigmas_rfx, sigma_eps], dim=-1)
        return targets

    def postprocess(
        self,
        proposed: dict[str, dict[str, torch.Tensor]],
        data: dict[str, torch.Tensor],
    ):
        '''reverse steps used in preprocessing for samples'''
        if 'samples' not in proposed['global']:
            return proposed

        if self.use_standardization:
            mean_y, mean_X, mean_Z = self.unpackMean(['y', 'X', 'Z']).values()
            std_y, std_X, std_Z = self.unpackStd(['y', 'X', 'Z']).values()

        # local postprocessing
        rfx_ = proposed['local']['samples'].clone()
        if self.use_standardization:
            b, m, q, s = rfx_.shape

            # standardize rfx
            rfx = rfx_ * std_y.view(b, 1, 1, 1)
            rfx[..., 1:, :] /= std_Z.view(b, 1, q - 1, 1)
            mean_Zb = (mean_Z.view(b, 1, q - 1, 1) * rfx[..., 1:, :]).sum(2)
            rfx[..., 0, :] = rfx_[..., 0, :] * std_y.view(b, 1, 1) - mean_Zb

            # patch samples
            rfx_ = rfx
        proposed['local']['samples'] = rfx_

        # global postprocessing
        samples = proposed['global']['samples'].clone()
        b, _, s = samples.shape
        d, q = self.d, self.q
        ffx_, sigmas_rfx_, sigma_eps_ = (
            samples[:, :d],
            samples[:, d:-1],
            samples[:, -1:],
        )

        # constrain stds to be positive
        if self.constrain:
            sigmas_rfx_ = maskedSoftplus(sigmas_rfx_)
            sigma_eps_ = maskedSoftplus(sigma_eps_)

        # analytical unstandardization
        if self.use_standardization:
            # unstandardize ffx
            onesies = data['d'] == 1
            ffx_[onesies, 0] = (
                0  # in pure intercept models the standardized intercept is 0
            )
            ffx = ffx_ * std_y.view(b, 1, 1)
            ffx[:, 1:] /= std_X.view(b, d - 1, 1)
            mean_Xb = (mean_X.view(b, d - 1, 1) * ffx[:, 1:]).sum(1)
            ffx[:, 0] = ffx_[:, 0] * std_y.view(b, 1) - mean_Xb + mean_y.view(b, 1)

            # unstandardize sigmas
            sigma_eps_ *= std_y.view(b, 1, 1)
            sigmas_rfx = sigmas_rfx_ * std_y.view(b, 1, 1)
            sigmas_rfx[:, 1:] /= std_Z.view(b, q - 1, 1)  # random slopes

            # sigma intercept with cov_sum
            ones = torch.ones_like(mean_X[..., 0:1])
            mean_Z1 = torch.cat([ones, mean_Z], dim=-1)
            weighted = rfx_.mean(-1) * mean_Z1.view(b, 1, q)
            cov = batchCovary(weighted, data['mask_m'])
            cov_sum = (cov.sum((-1, -2)) - cov[:, 0, 0]).unsqueeze(-1)
            sigma_0 = (sigmas_rfx[:, 0].square() - cov_sum).clamp(min=1e-12).sqrt()
            # ub = sigma_0.mean(-1).view(-1).topk(3)[0][-1]
            # sigmas_rfx[:, 0] = sigma_0.clamp(max=ub)
            sigmas_rfx[:, 0] = sigma_0

            # patch samples
            ffx_ = ffx.to('cpu')
            sigmas_rfx_ = sigmas_rfx.to('cpu')
            sigma_eps_ = sigma_eps_.to('cpu')

        proposed['global']['samples'] = torch.cat(
            [ffx_, sigmas_rfx_, sigma_eps_], dim=1
        )
        proposed['local']['samples'] = proposed['local']['samples'].to('cpu')
        return proposed

    def forward(
        self,
        data: dict[str, torch.Tensor],
        sample=False,
        n=(300, 200),
        log_prob=False,
        **kwargs,
    ) -> dict[str, torch.Tensor | dict]:
        # prepare
        proposed = {}
        inputs = self.inputs(data)
        b, m, _, _ = inputs.shape

        # local summaries
        summaries = self.summarizer_l(inputs, data['mask_n'])
        summaries = self.addMetadata(summaries, data, local=True)

        # global summary
        mask_m = None if self.training else data['mask_m']
        summary = self.summarizer_g(summaries, mask_m)
        context_g = self.addMetadata(summary, data, local=False)

        # global inference
        targets_g = self.targets(data, local=False)
        targets_g = self.preprocess(targets_g, data, local=False)
        loss, proposed['global'] = self.posterior_g(
            context_g, targets_g, sample=sample, n=n[0]
        )

        # local inference
        targets_l = self.targets(data, local=True)
        targets_l = self.preprocess(targets_l, data, local=True)
        if sample:
            global_params = proposed['global']['samples'].mean(-1).to(summaries.device)
        else:
            global_params = targets_g
        global_params = global_params.view(b, 1, -1).expand(b, m, -1)
        context_l = torch.cat([summaries, global_params], dim=-1)
        loss_l, proposed['local'] = self.posterior_l(
            context_l, targets_l, sample=sample, n=n[1]
        )

        # postprocessing
        proposed = self.postprocess(proposed, data)
        loss += loss_l.sum(-1) / data['m']
        return {'loss': loss, 'proposed': proposed}

    def estimate(self, data: dict[str, torch.Tensor], n=(300, 200)):
        with torch.no_grad():
            proposed = {}
            inputs = self.inputs(data)
            b, m, _, _ = inputs.shape
            mask_g = torch.cat(
                [
                    data['mask_d'],  # ffx
                    data['mask_q'],  # sigmas rfx
                    torch.ones(b, 1),  # sigma eps
                ],
                dim=-1,
            ).float()
            mask_l = data['mask_q'].unsqueeze(1).expand(b, m, -1).float()

            # summaries
            summaries = self.summarizer_l(inputs, data['mask_n'])
            summaries = self.addMetadata(summaries, data, local=True)
            summary = self.summarizer_g(summaries, data['mask_m'])

            # global inference
            context_g = self.addMetadata(summary, data, local=False)
            proposed['global'] = self.posterior_g.estimate(context_g, mask_g, n[0])

            # local inference
            global_params = proposed['global']['samples'].mean(-1)
            global_params = global_params.view(b, 1, -1).expand(b, m, -1)
            context_l = torch.cat([summaries, global_params], dim=-1)
            proposed['local'] = self.posterior_l.estimate(context_l, mask_l, n[1])

            # postprocessing
            proposed = self.postprocess(proposed, data)
        return proposed


