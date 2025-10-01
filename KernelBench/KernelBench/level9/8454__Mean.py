import torch
import torch.nn as nn
import torch.jit


class _Mean(nn.Module):

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return input.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
