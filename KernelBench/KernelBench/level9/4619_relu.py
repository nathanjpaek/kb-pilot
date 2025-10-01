import torch
import torch.nn as nn


class relu(nn.Module):

    def __init__(self, layer=10, channels=32):
        super(relu, self).__init__()
        layers = []
        for i in range(layer):
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
