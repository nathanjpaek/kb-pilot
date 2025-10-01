import torch
import torch.nn as nn


class Dense_block(nn.Module):
    """ This is the initial dense block as in the paper """

    def __init__(self, in_channels, out_channels):
        super(Dense_block, self).__init__()
        self.Dense = torch.nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform(self.Dense.weight.data, 1.0)
        self.activation = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.Dense(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
