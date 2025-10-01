import torch
from torch import nn


class Depth_Pointwise_Conv1d(nn.Module):

    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if k == 1:
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(in_channels=in_ch, out_channels=
                in_ch, kernel_size=k, groups=in_ch, padding=k // 2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=
            out_ch, kernel_size=1, groups=1)

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4, 'k': 4}]
