import torch
import torch.optim
import torch.nn as nn


class MMDLoss(nn.Module):

    def __init__(self):
        """
        Maximum Mean Discrepancy Loss
        """
        super(MMDLoss, self).__init__()
        self.eps = 1e-08

    def forward(self, f1: 'torch.Tensor', f2: 'torch.Tensor') ->torch.Tensor:
        loss = 0.0
        delta = f1 - f2
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
