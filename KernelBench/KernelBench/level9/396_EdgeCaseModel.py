import torch
import torch.nn as nn


class LayerWithRidiculouslyLongNameAndDoesntDoAnything(nn.Module):

    def forward(self, x):
        return x


class EdgeCaseModel(nn.Module):

    def __init__(self, throw_error=False, return_str=False):
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.model = LayerWithRidiculouslyLongNameAndDoesntDoAnything()

    def forward(self, x):
        x = self.conv1(x)
        x = self.model('string output' if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
