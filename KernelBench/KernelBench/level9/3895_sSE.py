import torch
import torch.nn as nn


class sSE(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=1,
            kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        x = self.pointwise(input_tensor)
        x = self.sigmoid(x)
        x = torch.mul(input_tensor, x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
