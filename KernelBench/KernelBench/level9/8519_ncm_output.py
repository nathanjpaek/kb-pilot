import torch
import torch.nn as nn


class ncm_output(nn.Module):

    def __init__(self, indim, outdim):
        super(ncm_output, self).__init__()
        self.linear = nn.Linear(indim, outdim)

    def forward(self, x):
        return -1 * torch.norm(x.reshape(x.shape[0], 1, -1) - self.linear.
            weight.transpose(0, 1).reshape(1, -1, x.shape[1]), dim=2).pow(2
            ) - self.linear.bias


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'indim': 4, 'outdim': 4}]
