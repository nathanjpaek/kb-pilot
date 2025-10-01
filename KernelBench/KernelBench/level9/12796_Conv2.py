import math
import torch
import torch.nn as nn


class Conv2(nn.Module):
    """ 1D conv with (kernel, stride)=(4, 2).

        Input:
            x: (N, 2L+2, in_channels) numeric tensor
            global_cond: (N, global_cond_channels) numeric tensor
        Output:
            y: (N, L, out_channels) numeric tensor
    """

    def __init__(self, in_channels, out_channels, global_cond_channels):
        super().__init__()
        ksz = 4
        self.out_channels = out_channels
        if 0 < global_cond_channels:
            self.w_cond = nn.Linear(global_cond_channels, 2 * out_channels,
                bias=False)
        self.conv_wide = nn.Conv1d(in_channels, 2 * out_channels, ksz, stride=2
            )
        wsize = 2.967 / math.sqrt(ksz * in_channels)
        self.conv_wide.weight.data.uniform_(-wsize, wsize)
        self.conv_wide.bias.data.zero_()

    def forward(self, x, global_cond):
        x1 = self.conv_wide(x.transpose(1, 2)).transpose(1, 2)
        if global_cond is not None:
            x2 = self.w_cond(global_cond).unsqueeze(1).expand(-1, x1.size(1
                ), -1)
        else:
            x2 = torch.zeros_like(x1)
        a, b = (x1 + x2).split(self.out_channels, dim=2)
        return torch.sigmoid(a) * torch.tanh(b)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4,
        'global_cond_channels': 4}]
