import torch
from torch import nn


class ClassWisePool(nn.Module):

    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        batch_size, num_channels, s = input.size()
        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, s)
        output = torch.sum(x, 2)
        return output.view(batch_size, num_outputs, s) / self.num_maps


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_maps': 4}]
