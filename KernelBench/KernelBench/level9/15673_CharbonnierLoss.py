import torch
import torch.nn as nn
from torch.nn import init as init


class CharbonnierLoss(nn.Module):

    def __init__(self, loss_weight=1.0, eps=1e-06):
        """
        the original eps is 1e-12
        """
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return torch.sum(torch.sqrt((pred - target) ** 2 + self.eps)
            ) / target.shape[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
