import torch
import torch.nn as nn


class HardSigmoid(nn.Module):
    """
    Hard Sigmoid
    """

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x.add_(0.5).clamp_(min=0, max=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
