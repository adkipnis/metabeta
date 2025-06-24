# implementation adapted from dingo https://github.com/dingo-gw/dingo
import torch
from torch import nn
from torchdiffeq import odeint
from metabeta.models.feedforward import MLP, ResidualNet
mse = nn.MSELoss(reduction='none')

class FlowMatching(nn.Module):
    def __init__(self,
                 d_theta: int,
                 d_context: int = 0,
                 d_hidden: int = 256,
                 n_blocks: int = 5,
                 alpha: float = 0., # exponent for sampling time with power law
                 activation: str = 'Mish',
                 dropout: float = 0.,
                 norm: str | None = None,
                 net_type: str = 'residual', #['mlp', 'residual']
                 **kwargs,
                 ):
        super().__init__()
        self.d_theta = d_theta
        self.d_context = d_context
        self.alpha = alpha

        if net_type == 'mlp':
            d_input = d_context + d_theta + 1 if d_context > 0 else d_theta + 1
            self.flow = MLP(
                    d_input=d_input,
                    d_output=d_theta,
                    d_hidden=(d_hidden,)*n_blocks,
                    activation=activation, dropout=dropout, norm=norm, skip=True)
            
        elif net_type == 'residual':
            d_input = d_context if d_context > 0 else d_theta + 1
            d_context_ = d_theta+1 if d_context > 0 else 0
            self.flow = ResidualNet(
                    d_input=d_input,
                    d_context=d_context_,
                    d_output=d_theta,
                    d_hidden=d_hidden,
                    n_blocks=n_blocks,
                    activation=activation, dropout=dropout, norm=norm)
        else:
            raise NotImplementedError('net_type must be mlp or residual')
            
        

    def forward(self, t: torch.Tensor, theta: torch.Tensor, context: torch.Tensor|None = None, mask=None) -> torch.Tensor:
        # separately embed timed theta and optionally context
        # pass both through continuous flow 
        t_theta = torch.cat([t, theta], dim=-1)
        if self.d_context > 0: # case: context model
            field = self.flow(context, context=t_theta)
        else: # case: context-less model
            field = self.flow(t_theta)
        if mask is not None:
            field = field * mask
        return field

    def sampleTime(self, batch_size: int) -> torch.Tensor:
        # sample t from a power-law distribution over [0, 1] (alpha=0 -> uniform)
        t = torch.rand(batch_size)
        return t.pow(1. / (1. + self.alpha))

    def loss(self, theta_1: torch.Tensor, context: torch.Tensor|None = None, mask=None) -> torch.Tensor:
        # sample time and theta_0
        b = len(theta_1)
        t = self.sampleTime(b).to(theta_1.device).unsqueeze(-1)
        theta_0 = torch.randn(b, self.d_theta, device=theta_1.device)
        if mask is not None:
            theta_0 = theta_0 * mask

        # get theta as convex combination
        theta = t * theta_1 + (1 - t) * theta_0

        # predict true vector field and calculate MSE loss
        field_true = theta_1 - theta_0
        field_pred = self.forward(t, theta, context, mask)
        loss = mse(field_true, field_pred)
        return loss

