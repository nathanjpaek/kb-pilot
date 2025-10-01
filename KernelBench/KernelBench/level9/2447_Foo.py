import torch
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data
import torch.optim
import torch.utils.data.distributed
import torch.autograd


class Foo(torch.nn.Module):

    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.n * input + self.m


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
