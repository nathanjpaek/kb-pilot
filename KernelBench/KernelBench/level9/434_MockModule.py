import torch
import torch.nn as nn
import torch.nn
import torch.optim


class MockModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=1)

    def forward(self, x: 'torch.Tensor'):
        return self.conv(x).mean(3).mean(2)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
