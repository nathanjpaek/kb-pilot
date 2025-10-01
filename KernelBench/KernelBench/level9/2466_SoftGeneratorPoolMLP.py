import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear


class SoftGeneratorPoolMLP(nn.Module):

    def __init__(self, nin, nhid1, nhid2, nout=1, bias=True):
        nn.Module.__init__(self)
        self.bias = bias
        self.linear1 = Linear(nin, nhid1, bias=self.bias)
        self.linear2 = Linear(nhid1, nhid2, bias=self.bias)
        self.linear3 = Linear(nhid2, nin, bias=self.bias)
    """
    def apply_bn(self, x): # Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()).cuda()
        return bn_module(x)
    """

    def forward(self, x):
        x = torch.mean(x, dim=0).unsqueeze(0)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nin': 4, 'nhid1': 4, 'nhid2': 4}]
