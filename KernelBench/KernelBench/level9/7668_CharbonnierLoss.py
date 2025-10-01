import torch
import torch.utils.data
from torch import nn
import torch.jit


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-06):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss / (c * b * h * w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
