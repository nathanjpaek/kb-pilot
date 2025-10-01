import torch
import torch.nn.functional as F
from torch import nn


class S_Loss(nn.Module):

    def __init__(self):
        super(S_Loss, self).__init__()

    def forward(self, x, label):
        loss = F.smooth_l1_loss(x, label)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
