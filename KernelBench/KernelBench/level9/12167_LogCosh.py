import torch
from torch import nn


def _assert_no_grad(tensor):
    assert not tensor.requires_grad


class LogCosh(nn.Module):

    def __init__(self, bias=1e-12):
        super().__init__()
        self.bias = bias

    def forward(self, output, target):
        _assert_no_grad(target)
        return torch.mean(torch.log(torch.cosh(target - output + self.bias)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
