import torch
import torch.nn as nn
from torch.optim import *


class RMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0.001).float()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1,
            keepdim=True)
        return torch.sqrt(loss / cnt)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
