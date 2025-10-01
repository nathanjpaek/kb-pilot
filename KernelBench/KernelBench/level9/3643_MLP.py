from torch.nn import Module
import torch
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn.init import xavier_uniform_


class MLP(Module):
    """ 
    Defines the NN model - in this case, there are 3 hidden layers,
    13 inputs (defined by data) in the 1st, 10 inputs in the second, 
    and 8 in the 3rd (with 1 output). The first two are activated by a sigmoid function,
    weighted by a xavier initalization scheme.

    """

    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        self.hidden3 = Linear(8, 6)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        self.hidden4 = Linear(6, 1)
        xavier_uniform_(self.hidden4.weight)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.hidden4(X)
        return X


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_inputs': 4}]
