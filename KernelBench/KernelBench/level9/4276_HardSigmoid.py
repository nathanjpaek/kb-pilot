import torch
import torch.nn as nn


class HardSigmoid(nn.Module):
    """hardsigmoid activation func used in squeeze-and-excitation module (see MobileNetV3)"""

    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return nn.ReLU6(inplace=True)(x + 3.0) / 6.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
