import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):

    def __init__(self, input_channels, output_channels, set_size, dim,
        particle_types):
        super().__init__()
        self.output_channels = output_channels
        self.set_size = set_size
        self.particle_types = particle_types
        self.linear1 = nn.Linear(input_channels, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear_fourvector = nn.Linear(dim, output_channels * set_size)
        self.linear_classification = nn.Linear(dim, set_size * particle_types)

    def forward(self, x):
        x1 = F.elu(self.linear1(x))
        x2 = F.elu(self.linear2(x1))
        vec = self.linear_fourvector(x2)
        vec = vec.view(vec.size(0), self.output_channels, self.set_size)
        particle = self.linear_classification(x2)
        particle = particle.view(particle.size(0), self.particle_types,
            self.set_size)
        return vec, particle


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4, 'set_size': 4,
        'dim': 4, 'particle_types': 4}]
