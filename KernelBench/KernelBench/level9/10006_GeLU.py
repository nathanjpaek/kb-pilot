import torch
import torch.nn as nn
import torch.nn.functional as F


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(0.7978845608 * (x + 0.044715 * x * x * x))
            )


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
