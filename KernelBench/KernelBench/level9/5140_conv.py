import torch
import torch.nn as nn
from torch.nn import init


class conv(nn.Module):
    """
    n*n conv with relu
    """

    def __init__(self, in_dim, out_dim, kernal_size, stride, padding):
        super(conv, self).__init__()
        self.con_layer = nn.Conv2d(in_dim, out_dim, kernal_size, stride,
            padding)
        self.relu = nn.ReLU(inplace=True)
        self.initi()

    def forward(self, input_):
        output = self.con_layer(input_)
        output = self.relu(output)
        return output

    def initi(self):
        init.normal_(self.con_layer.weight, std=0.01)
        if self.con_layer.bias is not None:
            init.constant_(self.con_layer.bias, 0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'kernal_size': 4, 'stride': 1,
        'padding': 4}]
