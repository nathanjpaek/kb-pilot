import torch
import torch.nn as nn


class softmax(nn.Module):

    def __init__(self, layer=10, channels=32):
        super(softmax, self).__init__()
        layers = []
        for i in range(layer):
            layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
