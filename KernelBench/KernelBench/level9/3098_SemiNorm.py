import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.jit
import torch.nn
from torch.nn.utils.spectral_norm import spectral_norm


class SemiNorm(nn.Module):

    def __init__(self, in_size, normalization=None):
        super().__init__()
        normalization = normalization or spectral_norm
        self.norm = nn.Linear(2 * in_size, in_size)
        self.bn = nn.LayerNorm(in_size)

    def forward(self, inputs):
        out = inputs.view(inputs.size(0), inputs.size(1), -1)
        mean = out.mean(dim=-1)
        std = out.std(dim=-1)
        out = self.bn(inputs)
        out = out.view(out.size(0), out.size(1), -1)
        features = self.norm(torch.cat((mean, std), dim=1))
        out = out + features.unsqueeze(-1)
        return out.view(inputs.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
