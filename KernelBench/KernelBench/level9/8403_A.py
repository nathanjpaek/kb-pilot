import torch
import torch.nn


class A(torch.nn.Module):

    def forward(self, x):
        return x + 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
