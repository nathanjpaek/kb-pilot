import torch
import torch.nn as nn


class Conv2dZeros(nn.Module):
    """Normal conv2d for reparameterize the latent variable.
    - weight and bias initialized to zero
    - scale channel-wise after conv2d
    """

    def __init__(self, in_channels, out_channels):
        super(Conv2dZeros, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=True)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x * torch.exp(self.scale * 3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
