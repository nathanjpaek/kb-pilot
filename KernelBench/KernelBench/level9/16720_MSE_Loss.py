import torch
import torch.nn as nn
from torch.nn import functional as F


class MSE_Loss(nn.Module):

    def __init__(self):
        super(MSE_Loss, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input, target, reduction='mean')


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
