import torch
import torch.nn as nn


class HINet(nn.Module):

    def __init__(self, in_ch):
        super(HINet, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(in_ch - in_ch // 2, affine=True)

    def forward(self, x):
        channels = x.shape[1]
        channels_i = channels - channels // 2
        x_i = x[:, :channels_i, :, :]
        x_r = x[:, channels_i:, :, :]
        x_i = self.instance_norm(x_i)
        return torch.cat([x_i, x_r], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4}]
