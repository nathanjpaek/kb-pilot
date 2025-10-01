import torch
from torch.utils.data import *
import torch.nn as nn


class lp_L2_Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, x, y):
        b = x.shape[0]
        loss = self.loss(x, y)
        return loss / b


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
