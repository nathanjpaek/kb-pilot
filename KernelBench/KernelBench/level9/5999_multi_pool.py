import torch
import torch.nn as nn


class multi_pool(nn.Module):

    def __init__(self):
        super(multi_pool, self).__init__()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool4 = nn.MaxPool2d(4, stride=2, padding=1)
        self.pool8 = nn.MaxPool2d(8, stride=2, padding=3)

    def forward(self, x):
        x1 = self.pool2(x)
        x2 = self.pool4(x)
        x3 = self.pool8(x)
        y = (x1 + x2 + x3) / 3.0
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
