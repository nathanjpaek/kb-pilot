import torch
import torch.nn as nn
import torch.distributed
import torch.distributions


class Clamp(nn.Module):

    def __init__(self, min=-1.0, max=1.0):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
