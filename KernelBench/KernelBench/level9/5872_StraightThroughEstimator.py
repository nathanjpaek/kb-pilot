from torch.autograd import Function
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import functional as F
import torch.jit


def straight_through_estimator(input: 'torch.Tensor') ->torch.Tensor:
    """ straight through estimator

    >>> straight_through_estimator(torch.randn(3, 3))
    tensor([[0., 1., 0.],
            [0., 1., 1.],
            [0., 0., 1.]])
    """
    return _STE.apply(input)


class _STE(Function):
    """ Straight Through Estimator
    """

    @staticmethod
    def forward(ctx, input: 'torch.Tensor') ->torch.Tensor:
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: 'torch.Tensor') ->torch.Tensor:
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):

    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, input: 'torch.Tensor'):
        return straight_through_estimator(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
