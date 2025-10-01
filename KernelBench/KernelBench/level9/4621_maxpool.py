import torch
import torch.nn as nn


class maxpool(nn.Module):

    def __init__(self, layer=10, channels=32):
        super(maxpool, self).__init__()
        layers = []
        for i in range(layer):
            layers.append(nn.MaxPool2d(3, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
