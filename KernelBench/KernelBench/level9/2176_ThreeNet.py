import torch
import torch.nn as nn


class ThreeNet(nn.Module):
    """
    A network with three layers. This is used for testing a network with more
    than one operation. The network has a convolution layer followed by two
    fully connected layers.
    """

    def __init__(self, input_dim: 'int', conv_dim: 'int', linear_dim: 'int'
        ) ->None:
        super(ThreeNet, self).__init__()
        self.conv = nn.Conv2d(input_dim, conv_dim, 1, 1)
        out_dim = 1
        self.pool = nn.AdaptiveAvgPool2d((out_dim, out_dim))
        self.linear1 = nn.Linear(conv_dim, linear_dim)
        self.linear2 = nn.Linear(linear_dim, 1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'conv_dim': 4, 'linear_dim': 4}]
