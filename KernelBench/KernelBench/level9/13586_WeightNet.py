import torch
from torch import nn as nn


class WeightNet(nn.Module):
    """WeightNet in Temporal interlace module.

    The WeightNet consists of two parts: one convolution layer
    and a sigmoid function. Following the convolution layer, the sigmoid
    function and rescale module can scale our output to the range (0, 2).
    Here we set the initial bias of the convolution layer to 0, and the
    final initial output will be 1.0.

    Args:
        in_channels (int): Channel num of input features.
        groups (int): Number of groups for fc layer outputs.
    """

    def __init__(self, in_channels, groups):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.groups = groups
        self.conv = nn.Conv1d(in_channels, groups, 3, padding=1)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        self.conv.bias.data[...] = 0

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        n, _, t = x.shape
        x = self.conv(x)
        x = x.view(n, self.groups, t)
        x = x.permute(0, 2, 1)
        x = 2 * self.sigmoid(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'groups': 1}]
