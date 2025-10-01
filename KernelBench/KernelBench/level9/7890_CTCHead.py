import torch
from torch import nn


class CTCHead(nn.Module):

    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=2)
        self.out_channels = out_channels

    def forward(self, x, labels=None):
        predicts = self.fc(x)
        if not self.training:
            predicts = self.softmax(predicts)
        return predicts


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
