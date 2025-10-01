import torch
import torch.nn as nn


class HardSwish(nn.Module):
    """
    Hard Swish
    """

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return x.add(0.5).clamp_(min=0, max=1).mul_(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
