import torch
from torch import nn


class Conv3dMaxPool(nn.Module):

    def __init__(self, out_channels: 'int', in_channels: 'int'):
        super().__init__()
        self.sat_conv3d = nn.Conv3d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.sat_maxpool = nn.MaxPool3d(3, stride=(1, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        x = self.sat_conv3d(x)
        return self.sat_maxpool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channels': 4, 'in_channels': 4}]
