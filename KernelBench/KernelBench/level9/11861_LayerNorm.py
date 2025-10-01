import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-06):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = Tensor(1, input_size).fill_(1)
        self.beta = Tensor(1, input_size).fill_(0)
        self.epsilon = epsilon
        if learnable:
            W = Parameter
        else:
            W = Variable
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).unsqueeze(1).expand_as(x)) / torch.sqrt(
            torch.var(x, 1).unsqueeze(1).expand_as(x) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
