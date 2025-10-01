import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional


class outblock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=2, output_padding=1):
        super(outblock, self).__init__()
        self.upconv = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.upconv.weight = nn.Parameter(Normal(0, 1e-05).sample(self.
            upconv.weight.shape))
        self.upconv.bias = nn.Parameter(torch.zeros(self.upconv.bias.shape))

    def forward(self, x):
        x = self.upconv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
