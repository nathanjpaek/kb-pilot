import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed


class AdaFM(nn.Module):

    def __init__(self, in_channel, out_channel, style_dim=0):
        super().__init__()
        self.style_gama = nn.Parameter(torch.ones(in_channel, out_channel, 
            1, 1))
        self.style_beta = nn.Parameter(torch.zeros(in_channel, out_channel,
            1, 1))

    def forward(self, input, style=0):
        h = input.shape[2]
        gamma = self.style_gama.repeat(1, 1, h, h)
        beta = self.style_beta.repeat(1, 1, h, h)
        out = gamma * input + beta
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
