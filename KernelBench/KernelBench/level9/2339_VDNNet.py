import torch
import torch.nn as nn


class VDNNet(nn.Module):

    def __init__(self):
        super(VDNNet, self).__init__()

    @staticmethod
    def forward(q_values):
        return torch.sum(q_values, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
