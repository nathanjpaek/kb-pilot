import torch
import torch.nn as nn


class Conv2d(nn.Module):

    def __init__(self):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(16, 33, kernel_size=1, padding=1, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 16, 64, 64])]


def get_init_inputs():
    return [[], {}]
