import torch
import torch.nn as nn


class BoundSoftmaxImpl(nn.Module):

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        max_x = torch.max(x, dim=self.axis).values
        assert self.axis == int(self.axis)
        x = torch.exp(x - max_x.unsqueeze(self.axis))
        s = torch.sum(x, dim=self.axis, keepdim=True)
        return x / s


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'axis': 4}]
