import torch
import torch.nn as nn


def _sn_to_specnorm(sn: 'int'):
    if sn > 0:

        def specnorm(module):
            return nn.utils.spectral_norm(module, n_power_iterations=sn)
    else:

        def specnorm(module, **kw):
            return module
    return specnorm


class EmbedNoise(nn.Module):

    def __init__(self, z_dim, channels, dim=4, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.pad = nn.Linear(z_dim, channels * dim * dim * dim)
        self.pad = specnorm(self.pad)
        self.nonlin = nn.LeakyReLU()
        self.z_dim = z_dim
        self.channels = channels
        self.dim = dim

    def forward(self, z):
        out = self.pad(z)
        out = self.nonlin(out)
        out = out.view((-1, self.channels, self.dim, self.dim, self.dim))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'channels': 4}]
