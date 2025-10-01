import torch
import torch.nn as nn


class HardSwish(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.0) / 6.0 * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
