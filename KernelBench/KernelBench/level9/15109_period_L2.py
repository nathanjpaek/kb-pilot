import torch
import numpy as np
import torch.nn as nn


def reduction_mean(loss):
    return loss.mean()


def reduction_none(loss):
    return loss


def reduction_sum(loss):
    return loss.sum()


class period_L2(nn.Module):

    def __init__(self, reduction='sum'):
        """
        periodic Squared Error
        """
        super().__init__()
        if reduction == 'sum':
            self.reduction = reduction_sum
        elif reduction == 'mean':
            self.reduction = reduction_mean
        elif reduction == 'none':
            self.reduction = reduction_none
        else:
            raise Exception('unknown reduction')

    def forward(self, theta_pred, theta_gt):
        dt = theta_pred - theta_gt
        loss = (torch.remainder(dt - np.pi / 2, np.pi) - np.pi / 2) ** 2
        assert (loss >= 0).all()
        loss = self.reduction(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
