import torch
import torch.nn as nn
from metabeta.utils import dInput, maskedMean, maskedStd

class Embedder(nn.Module):
    def __init__(self,
                 d_data: int,
                 d_model: int,
                 fx_type: str,
                 standardize: bool = False):
        super().__init__()
        self.d_data = d_data
        self.d_model = d_model
        self.d_input = dInput(d_data, fx_type)
        self.is_mfx = (fx_type == 'mfx')
        self.standardize = standardize
    
    def removeIntercept(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:]


class JointEmbedder(Embedder):
    ''' concatenates inputs before projecting '''
    def __init__(self,
                 d_data: int,
                 d_model: int,
                 fx_type: str,
                 standardize: bool = True):
        super().__init__(d_data, d_model, fx_type, standardize)
        self.emb = nn.Linear(self.d_input, d_model)
        
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != 0.)
        mean = maskedMean(x, mask)
        std = maskedStd(x, mask, mean)
        x = (x - mean) * mask / (std + 1e-12)
        return x, mean, std

    def forward(self, y: torch.Tensor, X: torch.Tensor,
                Z: None | torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        # assumes y [b, n, 1], x [b, n, d], z [b, n, d]
        if self.is_mfx:
            inputs = [y, X, Z]
        else:
            inputs = [y, X]
        inputs = torch.cat(inputs, dim=-1)
        out = self.emb(inputs)
        return out


class SeparateEmbedder(Embedder):
    ''' projects inputs separately and concatenates embeddings '''
    def __init__(self,
                 d_data: int,
                 d_model: int,
                 fx_type: str):
        super().__init__(d_data, d_model, fx_type)
        if self.is_mfx:
            d_out = d_model // 3
        else:
            d_out = d_model // 2
        self.emb_y = nn.Linear(1, d_out)
        self.emb_x = nn.Linear(d_data+1, d_out)
        if self.is_mfx:
            self.emb_z = nn.Linear(d_data+1, d_out)
            
    def forward(self, y: torch.Tensor, X: torch.Tensor,
                Z: None | torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        y_emb = self.emb_y(y)
        x_emb = self.emb_x(X)
        if self.is_mfx:
            z_emb = self.emb_z(Z)
            out = [y_emb, x_emb, z_emb]
        else:
            out = [y_emb, x_emb]
        return torch.cat(out, dim=-1)
    

# =============================================================================
if __name__ == "__main__":
    b = 100
    n = 50
    d = 8
    features = torch.randn(b, n, d)
    intercept = torch.ones((b, n, 1))
    X = torch.cat([intercept, features], dim=-1)
    y = torch.randn(b, n, 1)

    model = JointEmbedder(d, 128, 'ffx')
    output = model(y, X)

    model = SeparateEmbedder(d, 128, 'ffx')
    output = model(y, X)


