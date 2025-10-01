import torch
from torch import nn
from torch.nn import functional
import torch.autograd


class Loss(nn.Module):
    """A loss function."""


class PointwiseLoss(Loss):
    """Pointwise loss functions compute an independent loss term for each triple-label pair."""


class BCEAfterSigmoidLoss(PointwiseLoss):
    """A loss function which uses the numerically unstable version of explicit Sigmoid + BCE."""

    def __init__(self, reduction: 'str'='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: 'torch.FloatTensor', labels:
        'torch.FloatTensor', **kwargs) ->torch.FloatTensor:
        post_sigmoid = torch.sigmoid(logits)
        return functional.binary_cross_entropy(post_sigmoid, labels, **kwargs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
