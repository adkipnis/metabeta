from torch import nn

# --- activations
class GeGLU(nn.Module):
    ''' GELU-based Gated Linear Unit '''
    def forward(self, x): # 2d -> d
        x, gate = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)

ACTIVATIONS = {
    'ReLU': nn.ReLU, # default for NF
    'LeakyReLU': nn.LeakyReLU,
    'SiLU': nn.SiLU,
    'Mish': nn.Mish,
    'ELU': nn.ELU,
    'GELU': nn.GELU, # default for Transformers
    'GeGLU': GeGLU,
}

def getActivation(name: str) -> nn.Module:
    assert name in ACTIVATIONS, f'Unknown activation {name}'
    return ACTIVATIONS[name]()

