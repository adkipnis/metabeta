# implementation adapted from dingo https://github.com/dingo-gw/dingo
import torch
from torch import nn
from torch import vmap
from torchdiffeq import odeint
from metabeta.models.feedforward import MLP, ResidualNet
from tqdm import tqdm
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

    def _evalField(self, t: float, theta_t: torch.Tensor, context: torch.Tensor|None = None, mask=None):
        t_ = t * torch.ones(*theta_t.shape[:-1], device=theta_t.device).unsqueeze(-1)
        return self.forward(t_, theta_t, context, mask)

    def sample(self,
               n: int = 100,
               context: torch.Tensor|None = None,
               mask: torch.Tensor|None = None,
               method: str = 'euler',
               n_steps: int = 100,
               log_prob: bool = False):
        assert method in ['euler', 'rk4', 'dopri5'], 'ode integration method not supported'
        b = 1
        if context is not None:
            b = context.shape[0]
            context = context.unsqueeze(-2).expand(b, n, self.d_context)
        theta_0 = torch.randn(b, n, self.d_theta)
        if mask is not None:
            mask = mask.unsqueeze(-2).expand(b, n, self.d_theta)
            theta_0 = theta_0 * mask
        
        # log_prob
        log_q = None
        
        # integrate over ODE
        if method in ['dopri5']: # adaptive step size, slower and memory intensive
            t_ = torch.tensor([0., 1.]) # integration range
        elif method in ['euler', 'rk4']:
            t_ = torch.linspace(0., 1., n_steps+1) # integration steps
        else:
            raise ValueError
        trace = odeint(
            lambda t, theta_t: self._evalField(t, theta_t, context, mask),
            theta_0,
            t_, 
            atol=1e-5,
            rtol=1e-5,
            method=method, 
        )
        theta_1 = trace[-1]
        return theta_1, log_q

    def logProb(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# =============================================================================
if __name__ == '__main__':
    b, d, c = 50, 8, 15
    theta = torch.randn((b, d))
    context = torch.randn((b, c))
    
    # context-free model
    model = FlowMatching(d)
    loss = model.loss(theta)
    
    # context model
    model = FlowMatching(d, c)
    model.loss(theta, context).mean()

    # sample
    model.eval()
    samples = model.sample(n=100, context=context, method='dopri5')
    
    # mask
    theta[0,0] = 0.
    mask = (theta != 0.).float()
    loss = model.loss(theta, context, mask).sum() / mask.sum()
    samples = model.sample(n=100, context=context, mask=mask, method='dopri5')
    
    
    
