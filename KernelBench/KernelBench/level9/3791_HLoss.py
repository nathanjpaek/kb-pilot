import torch
from torch import nn
import torch.nn.functional as F


class HLoss(nn.Module):
    """
    Entropy loss used for entropy maximization.
    """

    def __init__(self, ignore_index=-1):
        super(HLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, labels):
        mask = (labels != self.ignore_index).float()
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * torch.matmul(mask, b.sum(dim=1))
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
