import torch
import torch.onnx
import torch.nn as nn


class AveragePool(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0,
            ceil_mode=True, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
