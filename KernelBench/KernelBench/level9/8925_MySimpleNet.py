import torch
import torch.nn.functional as F
from torch import nn


class MySimpleNet(nn.Module):
    """
        Very simple 2-layer net, slightly adapted from the docs:
            https://skorch.readthedocs.io/en/stable/user/quickstart.html
    """

    def __init__(self, num_in, num_feat, num_hidden=10, nonlin=F.relu):
        super(MySimpleNet, self).__init__()
        self.dense0 = nn.Linear(num_in, num_hidden)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_hidden, num_feat)
        self.output = nn.Linear(num_feat, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in': 4, 'num_feat': 4}]
