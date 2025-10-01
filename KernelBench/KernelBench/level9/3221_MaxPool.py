import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import torch.utils
import torch.cuda


class MaxPool(nn.Module):

    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')
        ):
        super(MaxPool, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride
        self.Max_Pool = nn.MaxPool2d(3, stride=stride, return_indices=True)
        self.pool_indices = None

    def forward(self, x):
        x_max = F.pad(x, [1] * 4)
        x_max, self.pool_indices = self.Max_Pool(x_max)
        return x_max


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'reduction': 4}]
