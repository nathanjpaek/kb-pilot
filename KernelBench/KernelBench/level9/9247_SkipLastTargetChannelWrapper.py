import torch
from torch import nn as nn
from torch.nn import MSELoss


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1
            ) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'
        target = target[:, :-1, ...]
        if self.squeeze_channel:
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


def get_inputs():
    return [torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'loss': MSELoss()}]
