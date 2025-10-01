import torch
from torch import nn
from torch.functional import F


class EmbedE(nn.Module):

    def __init__(self, l_in, l_h, l_g):
        super(EmbedE, self).__init__()
        self.fc = nn.Linear(l_in, l_h * l_g)

    def forward(self, h):
        h = F.relu(self.fc(h))
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'l_in': 4, 'l_h': 4, 'l_g': 4}]
