import torch
import torch.nn as nn
from torch.autograd import Variable


class tLN(nn.Module):

    def __init__(self, dimension, eps=1e-08, trainable=True):
        super(tLN, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1, 1),
                requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1, 1),
                requires_grad=False)

    def forward(self, inp):
        inp.size(0)
        mean = torch.sum(inp, 3, keepdim=True) / inp.shape[3]
        std = torch.sqrt(torch.sum((inp - mean) ** 2, 3, keepdim=True) /
            inp.shape[3] + self.eps)
        x = (inp - mean.expand_as(inp)) / std.expand_as(inp)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(
            x).type(x.type())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dimension': 4}]
