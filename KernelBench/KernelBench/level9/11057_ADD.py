import torch
import torch.nn as nn
import torch.utils.data


class ADD(nn.Module):

    def __init__(self, alpha=0.5):
        super(ADD, self).__init__()
        self.a = alpha

    def forward(self, x):
        return torch.add(x, self.a)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
