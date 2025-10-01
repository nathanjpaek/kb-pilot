from torch.nn import Module
import torch
import torch.nn.functional
from torch.nn import Parameter
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.nn.parallel
import torch.utils.data
import torch.optim
import torch.utils.data.distributed
from torch.nn import Module


class Model(Module):

    def __init__(self):
        super(Model, self).__init__()
        self.a = Parameter(torch.FloatTensor(4096 * 4096).fill_(1.0))
        self.b = Parameter(torch.FloatTensor(4096 * 4096).fill_(2.0))

    def forward(self, input):
        return input * self.a * self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 16777216])]


def get_init_inputs():
    return [[], {}]
