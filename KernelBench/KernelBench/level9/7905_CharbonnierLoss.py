import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):

    def __init__(self):
        """
        L1 Charbonnierloss.
        """
        super(CharbonnierLoss, self).__init__()

    def forward(self, x, y, eps=1e-06):
        diff = y - x
        error = torch.sqrt(diff * diff + eps)
        loss = torch.mean(error)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
