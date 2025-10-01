import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F
import torch.utils
import torch.cuda


class AvgPool(nn.Module):

    def __init__(self, in_channels, reduction, save_device=torch.device('cpu')
        ):
        super(AvgPool, self).__init__()
        self.save_device = save_device
        self.reduction = reduction
        if self.reduction:
            stride = 2
        else:
            stride = 1
        self.stride = stride
        self.Avg_Pool = nn.AvgPool2d(3, stride=stride)

    def forward(self, x):
        x_avg = F.pad(x, [1] * 4)
        x_avg = self.Avg_Pool(x_avg)
        return x_avg


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'reduction': 4}]
