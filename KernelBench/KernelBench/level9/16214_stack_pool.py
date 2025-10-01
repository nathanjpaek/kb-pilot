import torch
import torch.nn as nn


class stack_pool(nn.Module):

    def __init__(self):
        super(stack_pool, self).__init__()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool2s1 = nn.MaxPool2d(2, stride=1)
        self.pool3s1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.padding = nn.ReplicationPad2d((0, 1, 0, 1))

    def forward(self, x):
        x1 = self.pool2(x)
        x2 = self.pool2s1(self.padding(x1))
        x3 = self.pool3s1(x2)
        y = (x1 + x2 + x3) / 3.0
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
