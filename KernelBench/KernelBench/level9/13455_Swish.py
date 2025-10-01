import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.data
import torch.cuda
from torch.nn import Parameter
import torch.optim


class Swish(nn.Module):

    def __init__(self, dim):
        super(Swish, self).__init__()
        self.betas = Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x):
        pre_size = x.size()
        return x * nn.Sigmoid()(self.betas.view(-1, self.dim) * x.view(-1,
            self.dim)).view(pre_size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
