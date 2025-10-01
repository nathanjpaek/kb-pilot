import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class BiasAdd(nn.Module):

    def __init__(self, num_features):
        super(BiasAdd, self).__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))

    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
