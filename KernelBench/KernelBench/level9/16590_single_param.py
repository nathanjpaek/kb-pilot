import torch
import torch.nn as nn
import torch.distributions


class single_param(nn.Module):

    def __init__(self, value):
        super(single_param, self).__init__()
        self.p = nn.Parameter(torch.FloatTensor([value]))

    def forward(self):
        return torch.abs(self.p)


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'value': 4}]
