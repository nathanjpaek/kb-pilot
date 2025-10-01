import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from typing import *


class AngleLoss(nn.Module):

    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, angle, angle_hat):
        return torch.exp(F.mse_loss(angle_hat.float(), angle.float())) - 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
