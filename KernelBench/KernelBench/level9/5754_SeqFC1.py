import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqFC1(nn.Module):
    """ Neural network definition
    """

    def __init__(self, size):
        super(SeqFC1, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(in_features=self.size, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, coord):
        x = coord.float().view(coord.size(0), -1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
