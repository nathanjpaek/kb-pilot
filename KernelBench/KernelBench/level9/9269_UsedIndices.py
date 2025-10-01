import torch
from torch import nn
import torch.onnx


class UsedIndices(nn.Module):

    def __init__(self):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], ceil_mode
            =True, return_indices=True)

    def forward(self, x):
        y, indices = self.mp(x)
        return y - 42, indices + 42


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
