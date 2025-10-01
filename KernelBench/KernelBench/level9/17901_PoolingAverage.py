import torch
import torch.utils.data
import torch
import torch.nn as nn


class PoolingAverage(nn.Module):

    def __init__(self, input_dim=2048):
        super(PoolingAverage, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = input_dim

    def forward(self, x):
        x = torch.flatten(self.pool(x.view(x.size(0), x.size(1), x.size(2),
            -1)), 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
