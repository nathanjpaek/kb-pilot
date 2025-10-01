import torch
import torch.nn as nn


class conv_block(nn.Module):

    def __init__(self, init_shape):
        super(conv_block, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=init_shape[0], out_channels=
            init_shape[1], kernel_size=init_shape[2])
        self.relu = nn.ELU()
        nn.init.kaiming_uniform_(self.conv0.weight)

    def forward(self, input):
        out = self.relu(self.conv0(input))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'init_shape': [4, 4, 4]}]
