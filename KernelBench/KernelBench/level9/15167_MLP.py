from torch.nn import Module
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class MLP(Module):
    """
    A Simple two layers MLP to make SGC a bit better.
    """

    def __init__(self, nfeat, nhid, nclass, dp=0.2):
        super(MLP, self).__init__()
        self.W1 = nn.Linear(nfeat, nhid)
        self.W2 = nn.Linear(nhid, nclass)
        self.dp = dp
        self.act = nn.PReLU()
        self.num_class = nclass

    def forward(self, x):
        x = self.act(self.W1(x))
        x = nn.Dropout(p=self.dp)(x)
        return self.W2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4}]
