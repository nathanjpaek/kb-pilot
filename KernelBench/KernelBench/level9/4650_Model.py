from torch.nn import Module
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.optim
from torch.nn import Parameter
from torch.nn import Module


class Model(Module):

    def __init__(self):
        super(Model, self).__init__()
        self.x = Parameter(torch.FloatTensor(1, 4096 * 4096).fill_(1.0))

    def forward(self, input):
        return self.x * input


def get_inputs():
    return [torch.rand([4, 4, 4, 16777216])]


def get_init_inputs():
    return [[], {}]
