import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim


def hard_sigmoid(input_, inplace: 'bool'=False):
    """hard sigmoid function"""
    if inplace:
        return input_.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    return F.relu6(input_ + 3.0) / 6.0


class HardSigmoid(nn.Module):
    """hard sigmoid module"""

    def __init__(self, inplace: 'bool'=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_):
        return hard_sigmoid(input_, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
