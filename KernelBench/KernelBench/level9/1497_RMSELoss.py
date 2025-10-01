import torch
import torch as th
import torch.nn as nn


class RMSELoss(nn.Module):

    def __init__(self, reduction='mean', eps=1e-06):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, yhat, y):
        loss = th.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
