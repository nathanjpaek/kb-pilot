import torch
from torch import nn
import torch.utils.data
import torch.cuda
import torch.optim


class MaskedInstanceNorm1d(nn.Module):
    """Instance norm + masking."""
    MAX_CNT = 100000.0

    def __init__(self, d_channel: 'int', unbiased: 'bool'=True, affine:
        'bool'=False):
        super().__init__()
        self.d_channel = d_channel
        self.unbiased = unbiased
        self.affine = affine
        if self.affine:
            gamma = torch.ones(d_channel, dtype=torch.float)
            beta = torch.zeros_like(gamma)
            self.register_parameter('gamma', nn.Parameter(gamma))
            self.register_parameter('beta', nn.Parameter(beta))

    def forward(self, x: 'torch.Tensor', x_mask: 'torch.Tensor'
        ) ->torch.Tensor:
        """`x`: [B,C,T], `x_mask`: [B,T] => [B,C,T]."""
        x_mask = x_mask.unsqueeze(1).type_as(x)
        cnt = x_mask.sum(dim=-1, keepdim=True)
        cnt_for_mu = cnt.clamp(1.0, self.MAX_CNT)
        mu = (x * x_mask).sum(dim=-1, keepdim=True) / cnt_for_mu
        sigma = (x - mu) ** 2
        cnt_fot_sigma = (cnt - int(self.unbiased)).clamp(1.0, self.MAX_CNT)
        sigma = (sigma * x_mask).sum(dim=-1, keepdim=True) / cnt_fot_sigma
        sigma = (sigma + 1e-08).sqrt()
        y = (x - mu) / sigma
        if self.affine:
            gamma = self.gamma.unsqueeze(0).unsqueeze(-1)
            beta = self.beta.unsqueeze(0).unsqueeze(-1)
            y = y * gamma + beta
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_channel': 4}]
