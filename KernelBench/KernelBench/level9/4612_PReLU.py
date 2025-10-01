import torch
import torch.nn as nn


class PReLU(nn.Module):

    def __init__(self):
        super(PReLU, self).__init__()
        self.layer = nn.PReLU()

    def forward(self, x):
        x = self.layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
