import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleBody(nn.Module):

    def __init__(self, num_channels):
        super(SimpleBody, self).__init__()
        self.out_feats = 32
        self.fc1 = nn.Linear(num_channels, self.out_feats)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
