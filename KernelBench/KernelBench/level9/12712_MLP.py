from torch.nn import Module
import torch
from torch.nn import Linear
from torch.nn import Tanh
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class MLP(Module):
    """
    Summary: 1 hidden layer NN

    @param n_inputs (int): number of inputs in the current environment
    """

    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 40)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='tanh')
        self.act1 = Tanh()
        self.hidden2 = Linear(40, 40)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='tanh')
        self.act2 = Tanh()
        self.hidden3 = Linear(40, 4)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Tanh()

    def forward(self, X):
        """
        Summary: forward propagate input

        @param X (pytorch object): observation input batch (2d)

        @return X (pytorch object): input after all the neuralnet transofrmations, 
            i.e the NN estimation.
        """
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        return X


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_inputs': 4}]
