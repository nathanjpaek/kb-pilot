import torch
from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):

    def __init__(self, input_dim=512, output_dim=512, kernel_size=1,
        init_scale=1.0, no_weight_init=False):
        super(ConvEncoder, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size)
        if not no_weight_init:
            for layer in (self.conv,):
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

    def forward(self, x):
        _B, _D, _L = x.size()
        x = self.conv(x)
        x = F.relu(x)
        return x.flatten(1)


def get_inputs():
    return [torch.rand([4, 512, 64])]


def get_init_inputs():
    return [[], {}]
