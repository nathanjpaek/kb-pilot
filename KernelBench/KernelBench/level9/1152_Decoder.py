import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv3', nn.Conv2d(64, 32, 3, 1, 1))
        self.layers.add_module('Act3', nn.ReLU(inplace=True))
        self.layers.add_module('Conv4', nn.Conv2d(32, 16, 3, 1, 1))
        self.layers.add_module('Act4', nn.ReLU(inplace=True))
        self.layers.add_module('Conv5', nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, x):
        return self.layers(x)


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
