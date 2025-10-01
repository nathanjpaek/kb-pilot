import torch
from functools import partial
import torch.nn as nn


def dispatcher(dispatch_fn):

    def decorated(key, *args):
        if callable(key):
            return key
        if key is None:
            key = 'none'
        return dispatch_fn(key, *args)
    return decorated


@dispatcher
def activ_dispatch(activ):
    return {'none': nn.Identity, 'relu': nn.ReLU, 'lrelu': partial(nn.
        LeakyReLU, negative_slope=0.2)}[activ.lower()]


def spectral_norm(module):
    """ init & apply spectral norm """
    nn.init.xavier_uniform_(module.weight, 2 ** 0.5)
    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()
    return nn.utils.spectral_norm(module)


@dispatcher
def w_norm_dispatch(w_norm):
    return {'spectral': spectral_norm, 'none': lambda x: x}[w_norm.lower()]


class LinearBlock(nn.Module):
    """Pre-activate linear block"""

    def __init__(self, C_in, C_out, norm='none', activ='relu', bias=True,
        w_norm='none', dropout=0.0):
        super().__init__()
        activ = activ_dispatch(activ)
        if norm.lower() == 'bn':
            norm = nn.BatchNorm1d
        elif norm.lower() == 'none':
            norm = nn.Identity
        else:
            raise ValueError(
                f'LinearBlock supports BN only (but {norm} is given)')
        w_norm = w_norm_dispatch(w_norm)
        self.norm = norm(C_in)
        self.activ = activ()
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.linear = w_norm(nn.Linear(C_in, C_out, bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.linear(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in': 4, 'C_out': 4}]
