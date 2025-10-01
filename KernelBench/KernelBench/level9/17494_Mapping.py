import torch
import torch.nn as nn
import torch.fft


class Mapping(nn.Module):

    def __init__(self, z_size, out_size):
        super(Mapping, self).__init__()
        self.out_size = out_size
        self.mapping_layers = nn.ModuleList()
        self.linear = nn.Linear(z_size, z_size)
        self.relu = nn.ReLU(inplace=True)
        self.affine_transform = nn.Linear(z_size, out_size * 2)
        self.affine_transform.bias.data[:out_size] = 0
        self.affine_transform.bias.data[out_size:] = 1

    def forward(self, z):
        z = self.relu(self.linear(z))
        x = self.affine_transform(z)
        mean, std = torch.split(x, [self.out_size, self.out_size], dim=1)
        mean = mean[..., None, None]
        std = std[..., None, None]
        return mean, std


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'z_size': 4, 'out_size': 4}]
