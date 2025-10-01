import torch
import numpy as np
import torch.nn as nn


class Envelope(nn.Module):

    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.exponent = exponent
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs):
        env_val = 1 / inputs + self.a * inputs ** (self.p - 1
            ) + self.b * inputs ** self.p + self.c * inputs ** (self.p + 1)
        return torch.where(inputs < 1, env_val, torch.zeros_like(inputs))


class BesselBasisLayer(nn.Module):

    def __init__(self, num_radial, cutoff, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        freq_init = np.pi * torch.arange(1, num_radial + 1)
        self.frequencies = nn.Parameter(freq_init)

    def forward(self, inputs):
        d_scaled = inputs / self.cutoff
        d_scaled = torch.unsqueeze(d_scaled, -1)
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_radial': 4, 'cutoff': 4}]
