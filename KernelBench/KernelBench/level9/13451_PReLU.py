import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.data
import torch.cuda
from torch.nn import Parameter
import torch.optim


class PReLU(nn.Module):

    def __init__(self):
        super(PReLU, self).__init__()
        self.alpha = Parameter(torch.tensor(0.25))

    def forward(self, x):
        return nn.ReLU()(x) - self.alpha * nn.ReLU()(-x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
