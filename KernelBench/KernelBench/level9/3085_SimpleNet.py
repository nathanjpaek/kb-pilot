import torch
import torch.nn as nn
from torch.functional import F
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    Simple Neural Net model
    """

    def __init__(self):
        """
        Creates layers as class attributes.
        """
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        """
        Forward pass of the network.
        :param x:
        :return:
        """
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 2048])]


def get_init_inputs():
    return [[], {}]
