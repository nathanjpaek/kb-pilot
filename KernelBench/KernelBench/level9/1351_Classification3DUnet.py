import torch
import torch.nn as nn


class Classification3DUnet(nn.Module):

    def __init__(self, base_filters):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=base_filters, out_channels=1,
            kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        conv_c = self.conv(x)
        return conv_c


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'base_filters': 4}]
