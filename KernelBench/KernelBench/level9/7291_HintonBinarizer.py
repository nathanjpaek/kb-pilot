import torch
import torch.nn as nn
import torch.optim


class hinton_binarize(torch.autograd.Function):
    """
    Binarize function from the paper
    'SKIP RNN: LEARNING TO SKIP STATE UPDATES IN RECURRENT NEURAL NETWORKS'
    https://openreview.net/forum?id=HkwVAXyCW
    Works as round function but has a unit gradient:
    Binarize(x) := (x > 0.5).float()
    d Binarize(x) / dx := 1
    """

    @staticmethod
    def forward(ctx, x, threshold=0.5):
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HintonBinarizer(nn.Module):
    """
    Binarize function from the paper
    'SKIP RNN: LEARNING TO SKIP STATE UPDATES IN RECURRENT NEURAL NETWORKS'
    https://openreview.net/forum?id=HkwVAXyCW
    Works as round function but has a unit gradient:
    Binarize(x) := (x > 0.5).float()
    d Binarize(x) / dx := 1
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        return hinton_binarize.apply(x, threshold)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
