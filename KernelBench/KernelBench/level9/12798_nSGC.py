import math
import torch
import torch.nn.functional as F
import torch.utils.dlpack
import torch.nn as nn


class nSGC(nn.Module):

    def __init__(self, nfeat, nclass):
        super(nSGC, self).__init__()
        self.W1 = nn.Linear(nfeat, nclass * 2)
        self.W2 = nn.Linear(nclass * 2, nclass)
        self.init()

    def init(self):
        stdv = 1.0 / math.sqrt(self.W1.weight.size(1))
        self.W1.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nclass': 4}]
