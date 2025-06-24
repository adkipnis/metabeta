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

