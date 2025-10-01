import torch
from torch import nn


class SEBlock(nn.Module):

    def __init__(self, num_channels):
        super(SEBlock, self).__init__()
        self.lin1 = nn.Conv2d(num_channels, num_channels, 1)
        self.lin2 = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, x):
        h = nn.functional.avg_pool2d(x, int(x.size()[2]))
        h = torch.relu(self.lin1(h))
        h = torch.sigmoid(self.lin2(h))
        return x * h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
