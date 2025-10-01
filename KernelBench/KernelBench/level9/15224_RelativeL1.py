import torch
import torch.nn as nn


class RelativeL1(nn.Module):
    """ Relative L1 loss.
    Comparing to the regular L1, introducing the division by |c|+epsilon
    better models the human vision system’s sensitivity to variations
    in the dark areas. (where epsilon = 0.01, to prevent values of 0 in the
    denominator)
    """

    def __init__(self, eps=0.01, reduction='mean'):
        super().__init__()
        self.criterion = nn.L1Loss(reduction=reduction)
        self.eps = eps

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        base = y + self.eps
        return self.criterion(x / base, y / base)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
