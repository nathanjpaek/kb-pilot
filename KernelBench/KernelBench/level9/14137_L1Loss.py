import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class L1Loss(nn.Module):
    """ A simple mean absolute error (MAE) implementation.
    """

    def __init__(self, reduction='mean', **kwargs):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, **kwargs):
        return F.l1_loss(input, target, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
