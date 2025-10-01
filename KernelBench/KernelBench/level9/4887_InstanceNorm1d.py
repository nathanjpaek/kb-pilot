import torch
from torch import nn


class InstanceNorm1d(nn.Module):
    """
    Implementation of instance normalization for a 2D tensor of shape (batch size, features)
    """

    def __init__(self) ->None:
        super(InstanceNorm1d, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return (input - input.mean(dim=1, keepdim=True)) / input.std(dim=1,
            keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
