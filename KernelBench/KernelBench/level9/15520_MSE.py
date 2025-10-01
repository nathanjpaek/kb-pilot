import torch
import torch.nn as nn
from torch.optim import *


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0.001).float()
        loss = target * val_pixels - outputs * val_pixels
        return loss ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
