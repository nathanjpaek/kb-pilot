import torch
import torch.onnx
import torch.nn as nn


class MaxPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0,
            ceil_mode=True)

    def forward(self, x):
        return self.pool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
