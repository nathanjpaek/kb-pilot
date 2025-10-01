import torch
import torch.nn as nn
from typing import *


class SigmaL1SmoothLoss(nn.Module):

    def forward(self, output, target):
        reg_diff = torch.abs(target - output)
        reg_loss = torch.where(torch.le(reg_diff, 1 / 9), 4.5 * torch.pow(
            reg_diff, 2), reg_diff - 1 / 18)
        return reg_loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
