import torch
from torch import nn


class GaussianSubnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, tanh=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=1 if
            kernel > 1 else 0)
        self.activation = nn.Tanh() if tanh else nn.ReLU()
        if tanh:
            nn.init.xavier_normal_(self.conv.weight, gain=nn.init.
                calculate_gain('tanh'))
        else:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel': 4}]
