import torch
import torch.nn as nn


class Concat(nn.Module):

    def __init__(self, channels, **kwargs):
        super(Concat, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, equi_feat, c2e_feat):
        x = torch.cat([equi_feat, c2e_feat], 1)
        x = self.relu(self.conv(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
