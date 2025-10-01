import torch
from torch.nn.functional import softmax
from torch.nn import Linear
from torch.nn import Dropout
import torch.random


class Standard(torch.nn.Module):

    def __init__(self, in_features: 'int'):
        super().__init__()
        self.h1 = Linear(in_features, 50)
        self.d1 = Dropout()
        self.h2 = Linear(50, 50)
        self.d2 = Dropout()
        self.h3 = Linear(50, 50)
        self.d3 = Dropout()
        self.last_layer = Linear(50, 6)

    def preactivations(self, inputs: 'torch.Tensor'):
        x = torch.relu(self.h1(inputs))
        x = self.d1(x)
        x = torch.relu(self.h2(x))
        x = self.d2(x)
        x = torch.relu(self.h3(x))
        x = self.d3(x)
        return self.last_layer(x)

    def forward(self, inputs: 'torch.Tensor'):
        z = self.preactivations(inputs)
        return z, softmax(z)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
