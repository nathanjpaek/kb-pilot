import torch
import torch.nn as nn


def channel_squeeze(x, groups):
    """
    Channel squeeze operation.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, channels_per_group, groups, height, width).sum(dim=2)
    return x


class ChannelSqueeze(nn.Module):
    """
    Channel squeeze layer. This is a wrapper over the same operation. It is designed to save the number of groups.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """

    def __init__(self, channels, groups):
        super(ChannelSqueeze, self).__init__()
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_squeeze(x, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'groups': 1}]
