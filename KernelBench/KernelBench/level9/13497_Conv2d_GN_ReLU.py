import torch
import torch.nn as nn


class Conv2d_GN_ReLU(nn.Module):
    """ Implements a module that performs 
            conv2d + groupnorm + ReLU + 

        Assumes kernel size is odd
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1
        ):
        super(Conv2d_GN_ReLU, self).__init__()
        padding = 0 if ksize < 2 else ksize // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
            stride=stride, padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'num_groups': 1}]
