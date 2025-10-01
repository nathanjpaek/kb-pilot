import torch
import numpy as np
import torch.nn as nn


class period_L1(nn.Module):

    def __init__(self, reduction='sum'):
        """
        periodic Squared Error
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, theta_pred, theta_gt):
        dt = theta_pred - theta_gt
        dt = torch.abs(torch.remainder(dt - np.pi / 2, np.pi) - np.pi / 2)
        assert (dt >= 0).all()
        if self.reduction == 'sum':
            loss = dt.sum()
        elif self.reduction == 'mean':
            loss = dt.mean()
        elif self.reduction == 'none':
            loss = dt
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
