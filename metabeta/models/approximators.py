from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import distributions as D
from metabeta.models.embedders import Embedder, JointEmbedder, SeparateEmbedder
from metabeta.models.summarizers import Summarizer, DeepSet, PoolFormer
from metabeta.models.posteriors import Posterior, PointPosterior, MixturePosterior, DiscretePosterior, CouplingPosterior, MatchingPosterior, normalBins, halfNormalBins
mse = nn.MSELoss()

class Approximator(nn.Module):
    def __init__(self,
                 embedder: Embedder,
                 summarizer: Summarizer,
                 posterior: Posterior,
                 constrain: bool = True,
                 standardize: bool = False,
                 ):
        super().__init__()
        self.embedder = embedder
        self.summarizer = summarizer
        self.posterior = posterior
        self.constrain = constrain
        self.standardize = standardize
        self.embedder.standardize = standardize
    
    def targets(self, data: dict):
        raise NotImplementedError

    def forward(self, data: dict, sample: bool = False):
        raise NotImplementedError
        
    def addIntercept(self, summary: torch.Tensor, data: dict) -> torch.Tensor:
        return torch.cat([data['n'].unsqueeze(-1).sqrt(), summary], dim=-1)
    
    def addDataStats(self, summary: torch.Tensor) -> torch.Tensor:
        mu_y = self.embedder.mu_y.squeeze(1)
        sigma_y = self.embedder.sigma_y.squeeze(1)
        mu_X = self.embedder.mu_X.squeeze(1)
        sigma_X = self.embedder.sigma_X.squeeze(1)
        return torch.cat([mu_y, sigma_y, mu_X, sigma_X, summary], dim=-1)
    
    def moments(self, proposal: torch.Tensor | Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.posterior.other is not None:
            loc, scale = self.posterior.getLocScale(proposal['ffx'])
            loc_, scale_ = self.posterior.other.getLocScale(proposal['sigmas']) # type: ignore
            loc = torch.cat([loc, loc_], dim=-1)
            scale = torch.cat([scale, scale_], dim=-1)
            return loc, scale
        else:
            return self.posterior.getLocScale(proposal)
        
    def quantiles(self, proposal: torch.Tensor | Dict[str, torch.Tensor],
                  roots: list = [.025, .50, .975]) -> torch.Tensor:
        if self.posterior.other is not None:
            quantiles = self.posterior.getQuantiles(proposal['ffx'], roots)
            quantiles_ = self.posterior.other.getQuantiles(proposal['sigmas'], roots) # type: ignore
            quantiles = torch.cat([quantiles, quantiles_], dim=1)
            return quantiles
        else:
            return self.posterior.getQuantiles(proposal, roots)

    def examples(self, indices: list, batch: dict, proposal: torch.Tensor | Dict[str, torch.Tensor],
                 printer: Callable, console_width: int) -> None:
        if len(indices) == 0:
            return 
        ffx = batch['ffx']
        d = ffx.shape[1]
        sigmas_rfx = batch['sigmas_rfx'] if 'sigmas_rfx' in batch else None
        ffx_a = batch['analytical']['ffx']['mu'] if 'analytical' in batch else None
        loc_ = None
        if isinstance(proposal, dict) and 'global' in proposal:
            if 'local' in proposal:
                loc_, _ = self.posterior.getLocScale(proposal['local'])
            proposal = proposal['global']
        loc, scale = self.moments(proposal)
        for i in indices:
            n_i = batch['n'][i]
            sigma_i = batch['sigma_error'][i]
            mask = (ffx[i] != 0.)
            beta_i = ffx[i, mask].detach().numpy()
            mean_i = loc[i, :d][mask].detach().numpy()
            std_i = scale[i, :d][mask].detach().numpy()
            sigma_i_ = loc[i, d].detach().numpy()
            printer(f"\n{console_width * '-'}")
            printer(f"n={n_i}, sigma={sigma_i:.2f}, predicted={sigma_i_:.2f}")
            # FFX
            printer(f"True FFX   : {beta_i}")
            if ffx_a is not None:
                printer(f"Optimal FFX: {ffx_a[i, mask].detach().numpy()}")
            printer(f"Mean FFX   : {mean_i}")
            printer(f"SD FFX     : {std_i}")
            # Variances
            if sigmas_rfx is not None:
                mask_rfx = (sigmas_rfx[i] != 0.)
                sigmas_rfx_i = sigmas_rfx[i][mask_rfx].detach().numpy()
                sigmas_rfx_i_ = loc[i, d+1:][mask_rfx].detach().numpy()
                printer(f"True sigmas RFX: {sigmas_rfx_i}")
                printer(f"Mean sigmas RFX: {sigmas_rfx_i_}")
            # Random Effects
            if loc_ is not None:
                rfx_i = batch['rfx'][i][..., mask_rfx][:5, 0].detach().numpy()
                rfx_i_ = loc_[i][:, mask_rfx][:5, 0].detach().numpy()
                printer(f"True random intercept (first 5): {rfx_i}")
                printer(f"Mean random intercept (first 5): {rfx_i_}")
            printer(f"{console_width * '-'}\n")

    def plotRecovery(self, ds: dict, proposed: torch.Tensor,
                     show_error: bool = False, color: str = 'darkgreen', alpha: float = 0.2) -> None:
        targets, names = self.targets(ds)
        means, stds = self.moments(proposed)
        min_val = -10.5
        max_val = 10.5
        mask = (targets != 0.)
        D = len(names)
        assert means.shape[-1] == mask.shape[-1] == D, "shape mismatch"
        w = int(torch.tensor(D).sqrt().ceil())
        _, axs = plt.subplots(figsize=(8 * w, 7 * w), ncols=w, nrows=w)
        axs = axs.flatten()
        for i in range(D):
            # prepare
            ax = axs[i]
            mask_i = mask[..., i]
            if mask_i.sum() == 0:
                axs[i].set_visible(False)
                continue
            targets_i = targets[mask_i, i]
            mean_i = means[mask_i, i]
            std_i = stds[mask_i, i]
            table_i = torch.cat(
                [targets_i.unsqueeze(-1), mean_i.unsqueeze(-1)],
                dim=-1).permute(1,0)
            r = torch.corrcoef(table_i)[0,1]
            bias = (targets_i - mean_i).mean()
            rmse = mse(targets_i, mean_i).sqrt()
            
            if i == ds['X'].shape[-1]:
                min_val = -0.5
                max_val = 4.5

            # plot
            ax.set_axisbelow(True)
            ax.grid(True)
            ax.plot([min_val, max_val], [min_val, max_val],
                    '--', lw=2, zorder=1, color='grey')
            if show_error:
                ax.errorbar(targets_i, mean_i, yerr=std_i,
                            fmt='', alpha=0.3, color='grey',
                            capsize=0, linestyle='none')
            ax.scatter(targets_i, mean_i,
                       alpha=alpha, color=color, label=names[i])
            ax.text(
                0.75, 0.1,
                f'r = {r.item():.3f}\nBias = {bias.item():.3f}\nRMSE = {rmse.item():.3f}',
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=16,
                bbox=dict(boxstyle='round',
                          facecolor=(1, 1, 1, 0.7),
                          edgecolor=(0, 0, 0, alpha),
                          ),
            )
            ax.set_xlabel('true', fontsize=20)
            ax.set_ylabel('estimated', fontsize=20)
            ax.legend()
 
        for i in range(D, w*w):
            axs[i].set_visible(False)
        
        
    def plotRecoveryLocal(self, ds: dict, proposed: torch.Tensor, show_error: bool = False) -> None:
        targets = ds['rfx']
        b, m, d = targets.shape
        names = [rf'$b_{i}$' for i in range(d)]
        means, stds = self.moments(proposed)
        min_val = torch.tensor([targets.min(), means.min()]).min()
        max_val = torch.tensor([targets.max(), means.max()]).max()
        mask = (targets != 0.)
        w = int(torch.tensor(d).sqrt().ceil())
        _, axs = plt.subplots(figsize=(8 * w, 7 * w), ncols=w, nrows=w)
        axs = axs.flatten()
        for i in range(d):
            # prepare
            ax = axs[i]
            mask_i = mask[..., i]
            if mask_i.sum() == 0:
                axs[i].set_visible(False)
                continue
            targets_i = targets[mask_i][:, i]
            mean_i = means[mask_i][:, i]
            std_i = stds[mask_i][:, i]
            table_i = torch.cat(
                [targets_i.unsqueeze(-1), mean_i.unsqueeze(-1)],
                dim=-1).permute(1,0)
            r = torch.corrcoef(table_i)[0,1]
            bias = (targets_i - mean_i).mean()
            rmse = mse(targets_i, mean_i).sqrt()
    
            # plot
            idx = torch.randperm(len(mean_i))[:len(mean_i)//m]
            mean_i = mean_i[idx]
            targets_i = targets_i[idx]
            
            alpha = 0.2
            ax.set_axisbelow(True)
            ax.grid(True)
            ax.plot([min_val, max_val], [min_val, max_val],
                    '--', lw=2, zorder=1, color='grey')
            if show_error:
                ax.errorbar(targets_i, mean_i, yerr=std_i,
                            fmt='', alpha=0.3, color='grey',
                            capsize=0, linestyle='none')
            ax.scatter(targets_i, mean_i,
                       alpha=alpha, color='purple', label=names[i])
            ax.text(
                0.75, 0.1,
                f'r = {r.item():.3f}\nBias = {bias.item():.3f}\nRMSE = {rmse.item():.3f}',
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=16,
                bbox=dict(boxstyle='round',
                          facecolor=(1, 1, 1, 0.7),
                          edgecolor=(0, 0, 0, alpha),
                          ),
            )
            ax.set_xlabel('true', fontsize=20)
            ax.set_ylabel('estimated', fontsize=20)
            ax.legend()
 
        for i in range(d, w*w):
            axs[i].set_visible(False)

class ApproximatorFFX(Approximator):
    def __init__(self,
                 embedder: nn.Module,
                 summarizer: Summarizer,
                 posterior: Posterior,
                 constrain: bool = True, # constrains sigma
                 standardize: bool = False, # standardizes inputs
                 ):
        super().__init__(embedder, summarizer, posterior, constrain, standardize)
        

    @classmethod
    def build(cls,
              d_data: int, # input dimension
              d_hidden: int, # projection dimension
              d_ff: int, # feedforward dimension (transformer and mlps)
              d_out: int, # summary dimension
              dropout: float = 0.01,
              activation: str = 'ReLU',
              n_heads: int = 4, n_blocks: int = 1,
              emb_type: str = 'joint',
              sum_type: str = 'deepset',
              post_type: str = 'mixture',
              bins: int = 100, components: int = 1, flows: int = 3,
              max_m: int = 30,
              standardize: bool = True):

        # 1. embedder
        if emb_type == 'joint':
            embedder = JointEmbedder(d_data, d_hidden, 'ffx')
        elif emb_type == 'separate':
            embedder = SeparateEmbedder(d_data, d_hidden, 'ffx')
        else:
            raise ValueError(f'embedding type {emb_type} unknown')

        # 2. summarizer
        if sum_type == 'deepset':
            summarizer = DeepSet(
                d_model=d_hidden, d_ff=d_ff, d_output=d_out,
                n_blocks=n_blocks, dropout=dropout, activation=activation)
        elif sum_type == 'poolformer':
            summarizer = PoolFormer(
                d_model=d_hidden, d_ff=d_ff, d_output=d_out, depth=2, n_heads=n_heads, 
                n_blocks=n_blocks, dropout=dropout, activation='GELU')
        else:
            raise ValueError(f'transformer type {sum_type} unknown')

        # 3. posteriors
        d_ffx = d_data + 1 # slopes + bias
        d_out += 1 # add dataset size as additional summary variable
        if standardize:
            d_out += 2 * d_ffx # add moments of y and X
        if post_type == 'point':
            posterior = PointPosterior(d_out, d_ffx + 1)
        elif post_type == 'mixture':
            posterior = MixturePosterior(
                d_out, d_ffx, n_components=components, comp_dist=D.Normal)
            posterior.other = MixturePosterior(
                d_out, 1, n_components=components, comp_dist=D.LogNormal) 
        elif post_type == 'discrete':
            borders = normalBins(3., steps=bins+1)
            posterior = DiscretePosterior(d_out, d_ffx, borders)
            borders_ = halfNormalBins(3., steps=bins+1)
            posterior.other = DiscretePosterior(d_out, 1, borders_)
        elif 'flow' in post_type:
            if post_type in ['flow-affine', 'flow-spline']:
                net_kwargs = {'d_hidden': d_ff, 'n_blocks': n_blocks,
                              'activation': activation, 'dropout': dropout}
                transform = 'affine' if post_type == 'flow-affine' else 'rq'
                posterior = CouplingPosterior(
                    d_out, d_ffx+1, n_flows=flows, transform=transform,
                    net_kwargs=net_kwargs, fx_type='ffx')
            elif post_type == 'flow-matching':
                net_kwargs = {'d_hidden': 256, 'n_blocks': 5,
                              'activation': activation, 'dropout': dropout,
                              }
                posterior = MatchingPosterior(
                    d_out, d_ffx+1, net_kwargs=net_kwargs, fx_type='ffx')
            else:
                raise ValueError
            posterior.other = None
        else:
            raise ValueError(f'posterior type {post_type} unknown')
        return cls(embedder, summarizer, posterior, standardize=standardize)
    
    def targets(self, data):
        out = torch.cat([data['ffx'], data['sigma_error'].unsqueeze(-1)], dim=-1)
        names = [rf'$\beta_{i}$' for i in range(data['ffx'].shape[1])] + [r'$\sigma_e$'] 
        return out, np.array(names)
    
    def _standardize(self, targets: torch.Tensor):
        mu_y = self.embedder.mu_y.squeeze()
        sigma_y = self.embedder.sigma_y.squeeze()
        mu_X = self.embedder.mu_X.squeeze(1)
        sigma_X = self.embedder.sigma_X.squeeze(1) + 1e-12
        
        # sigma
        log_sigma = targets[:, -1]
        targets[:, -1] = log_sigma - sigma_y.log()
        
        # slopes
        beta = targets[:, 1:-1]
        beta_std = beta * sigma_X / sigma_y.unsqueeze(-1)
        targets[:, 1:-1] = beta_std
            
        # intercept
        beta0 = targets[:, 0]
        sum_term = (beta_std * mu_X / sigma_X).sum(1)
        targets[:, 0] = (beta0 - mu_y) / sigma_y - sum_term
        return targets
    
    def _unstandardize(self, samples: torch.Tensor):
        mu_y = self.embedder.mu_y.squeeze(1)
        sigma_y = self.embedder.sigma_y.squeeze(1)
        mu_X = self.embedder.mu_X.squeeze(1).unsqueeze(-1)
        sigma_X = self.embedder.sigma_X.squeeze(1).unsqueeze(-1) + 1e-12
        
        # sigma
        sigma_log_std = samples[:, -1]
        samples[:, -1] = sigma_log_std + sigma_y.log()

        # slopes
        beta_std = samples[:, 1:-1].clone()
        samples[:, 1:-1] = beta_std / sigma_X * sigma_y.unsqueeze(1)
        
        # intercept
        beta0_std = samples[:, 0]
        sum_term = (beta_std * mu_X / sigma_X).sum(1)
        samples[:, 0] = (beta0_std + sum_term) * sigma_y + mu_y
        return samples
        
    
    def preprocess(self, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # unconstrain sigma
        if self.constrain:
            sigma = targets[:, -1]
            targets[:, -1] = sigma.log()
        # standardize
        # if self.standardize:
        #     targets = self._standardize(targets)
        return targets
    
    def forward(self, data, sample=False, log_prob=False, **kwargs):
        h = self.embedder(**data) # (b, n, d_hidden)
        summary = self.summarizer(h, data['mask_n']) # (b, d_out)
        if self.posterior.other is None:
            targets, _ = self.targets(data)
            loss, proposed = self.posterior(
                summary, targets, sample=sample, log_prob=log_prob, constrain=True, **kwargs)
        else:
            targets1 = data['ffx'] 
            targets2 = data['sigma_error'].unsqueeze(-1)
            loss, proposed = self.posterior(summary, targets1, sample=sample)
            loss_, proposed_ = self.posterior.other(summary, targets2, sample=sample, **kwargs) # type: ignore
            loss = loss + loss_
            proposed = {'ffx': proposed, 'sigmas': proposed_}
        return loss, {'global': proposed}, summary


# class ApproximatorMFX(Approximator):
#     def __init__(self,
#                  embedder: nn.Module,
#                  summarizer: Summarizer,
#                  posterior_g: Posterior,
#                  posterior_l: Posterior,
#                  ):
#         super().__init__(embedder, summarizer, posterior_g)
#         self.posterior_l = posterior_l

#     @classmethod
#     def build(cls,
#               d_data: int, d_hidden: int, d_ff: int, d_out: int,
#               dropout: float = 0.1, activation: str = 'GELU',
#               n_heads: int = 4, n_blocks: int = 1,
#               emb_type: str = 'joint',
#               sum_type: str = 'deepset',
#               post_type: str = 'mixture',
#               bins: int = 100, components: int = 1, flows: int = 3,
#               max_m: int = 30):

#         # 1. embedder
#         if emb_type == 'joint':
#             embedder = JointEmbedder(d_data, d_hidden, 'mfx')
#         elif emb_type == 'separate':
#             embedder = SeparateEmbedder(d_data, d_hidden, 'mfx')
#         else:
#             raise ValueError(f'embedding type {emb_type} unknown')

#         # 2. summarizer
#         if sum_type == 'deepset':
#             summarizer = DeepSet(d_hidden, d_ff, d_out, n_blocks, dropout, activation)
#         elif sum_type == 'poolformer':
#             summarizer = PoolFormer(d_hidden, d_ff, d_out, n_heads, n_blocks, dropout, activation)
#         else:
#             raise ValueError(f'transformer type {sum_type} unknown')

#         # 3. posteriors
#         d_fx = 1 + d_data # bias + slopes (ffx or rfx)
#         d_var = 1 + d_fx # error + rfx vars
#         d_glob = d_out * max_m
#         d_loc = d_out + d_fx + d_var
#         if post_type == 'point':
#             posterior_g = PointPosterior(d_glob, d_fx + d_var)
#             posterior_l = PointPosterior(d_loc, d_fx)
#         elif post_type == 'mixture':
#             posterior_g = MixturePosterior(d_glob, d_fx, n_components=components, comp_dist=D.Normal)
#             posterior_g.other = MixturePosterior(d_hidden + d_fx, d_var, n_components=components, comp_dist=D.LogNormal) 
#             posterior_l = MixturePosterior(d_loc, d_fx, n_components=components, comp_dist=D.Normal)
#         elif post_type == 'discrete':
#             posterior_g = DiscretePosterior(d_glob, d_fx, bins=normalBins(5., steps=bins+1))
#             posterior_g.other = DiscretePosterior(d_hidden + d_fx, d_var, bins=halfNormalBins(5., steps=bins+1))
#             posterior_l = DiscretePosterior(d_loc, d_fx, bins=normalBins(5., steps=bins+1))
#         elif post_type == 'flow-a':
#             posterior_g = FlowPosterior(d_glob, d_fx + d_var, n_flows=flows, transform='affine')
#             posterior_l = FlowPosterior(d_loc, d_fx, n_flows=flows, transform='affine')
#         else:
#             raise ValueError(f'posterior type {post_type} unknown')
        
#         return cls(embedder,
#                    summarizer,
#                    posterior_g,
#                    posterior_l,
#                    )

#     def targets(self, data):
#         targets_ = [
#             data['ffx'],
#             data['sigma_error'].unsqueeze(-1),
#             data['sigmas_rfx']
#             ]
#         names = (
#             [rf'$\beta_{i}$' for i in range(data['ffx'].shape[1])] +
#             [r'$\sigma_e$'] + 
#             [rf'$\sigma_{i}$' for i in range(data['sigmas_rfx'].shape[1])] 
#             )
#         return torch.cat(targets_, dim=-1), np.array(names)

#     def forward(self, data: dict, sample=True, local=True, n=1000):
#         h = self.embedder(**data) # (b, m, n_i, d_hidden)
#         summaries = self.summarizer(h, data['mask_n']) # (b, m, d_hidden)
#         b, m = summaries.shape[:2]
#         summary = summaries.view(b, -1)
        
#         # b, m, n, d_hidden = h.shape
#         # h_ = h.view(b, m*n, d_hidden)
#         # mask = data['mask_n'].view(b, m*n)
#         # summary = self.summarizer(h_, mask)
        
#         # global inference
#         targets, _ = self.targets(data)
#         if self.posterior.other is None:
#             loss, proposed = self.posterior(summary, targets, sample=sample, n=n, constrain=True)
#         else:
#             targets0 = data['ffx']
#             context0 = summary
#             loss0, proposed0 = self.posterior(context0, targets0, sample=sample)
            
#             targets1 = targets[..., targets0.shape[1]:]
#             context1 = torch.cat([self.reviser(summary), targets0], dim=-1)
#             loss1, proposed1 = self.posterior.other(context1, targets1, sample=sample, constrain=True) # type: ignore
#             loss = loss0 + loss1
#             proposed = dict(ffx=proposed0, sigmas=proposed1)

#         # local inference
#         if local:
#             context_ = targets.unsqueeze(1).expand((b,m,-1))
#             context = torch.cat([summaries, context_], dim=-1)
#             loss_, proposed_ = self.posterior_l(context, data['rfx'], sample=sample, n=100)
#             loss = loss + loss_.sum(-1) / data['m']
#             proposed = {'global': proposed, 'local': proposed_}
#         return loss, proposed, summary

    #def predict(self, data):
    # if sample: # this only concerns normalizing flows
    #     s = proposed.shape[-1]
    #     proposed_ = proposed.permute(0,2,1)[..., :-1].unsqueeze(2).expand((b,s,m,-1))            
    #     summary_l_ = summaries.unsqueeze(1).expand((b,s,m,-1))
    #     context = torch.cat([summary_l_, proposed_], dim=-1)
    #     rfx = data['rfx'].unsqueeze(1).expand((b,s,m,-1))   
    #     _, proposed_ = self.posterior_l(context, rfx, sample=True, n=1)

# =============================================================================
if __name__ == "__main__":
    # data
    b = 100
    n = 50
    d = 8
    X = torch.randn(b, n, d)
    targets = torch.randn(b, d)
    sigma_error = torch.randn(b,).abs()
    y = torch.randn(b, n, 1)
    data = dict(X=X, y=y, ffx=targets, sigma_error=sigma_error, mask_n=None)


    # build
    model = ApproximatorFFX.build(d-1, 128, 256, 32,
                               emb_type='joint',
                               sum_type='deepset',
                               post_type='flow-affine', components=3)
    loss, proposed, summary = model(data)


