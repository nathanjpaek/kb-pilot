import torch
from torch import Tensor
from torch import nn


class SmoothSoftmax(nn.Module):

    def forward(self, x: 'Tensor'):
        logistic_value = torch.sigmoid(x)
        return logistic_value / logistic_value.sum(dim=-1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
