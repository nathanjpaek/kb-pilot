import torch
import torch.nn as nn


class HardSwish(nn.Module):
    """hardswish activation func (see MobileNetV3)"""

    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return x * nn.ReLU6(inplace=True)(x + 3.0) / 6.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
