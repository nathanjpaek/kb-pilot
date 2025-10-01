import torch
from torch import nn
import torch.onnx


class UnusedIndices(nn.Module):

    def __init__(self):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], ceil_mode
            =True)

    def forward(self, x):
        return self.mp(x) - 42


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
