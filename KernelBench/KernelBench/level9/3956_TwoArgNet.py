import torch
from torch import nn


class TwoArgNet(nn.Module):

    def __init__(self, inc, outc):
        super().__init__()
        self.layer = nn.Linear(inc, outc)

    def forward(self, t1, t2):
        return self.layer(torch.cat((t1, t2), dim=1)).sigmoid()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inc': 4, 'outc': 4}]
