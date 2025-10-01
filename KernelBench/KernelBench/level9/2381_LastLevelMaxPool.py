import torch
from torchvision.transforms import functional as F
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class LastLevelMaxPool(nn.Module):

    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
