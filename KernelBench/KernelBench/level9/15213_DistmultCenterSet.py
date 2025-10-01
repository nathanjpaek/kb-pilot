import torch
import torch.nn as nn
import torch.nn.functional as F


class DistmultCenterSet(nn.Module):

    def __init__(self, dim, aggr=torch.max, nonlinear=True):
        super(DistmultCenterSet, self).__init__()
        self.dim = dim
        self.layers = nn.Parameter(torch.zeros(self.dim * 4 + 4, self.dim))
        nn.init.xavier_uniform_(self.layers[:self.dim * 4, :])
        self.aggr = aggr
        self.nonlinear = nonlinear

    def forward(self, embeddings):
        w1, w2, w3, w4, b1, b2, b3, b4 = torch.split(self.layers, [self.dim
            ] * 4 + [1] * 4, dim=0)
        x = F.relu(F.linear(embeddings, w1, b1.view(-1)))
        x = F.linear(x, w2, b2.view(-1))
        if self.nonlinear:
            x = F.relu(x)
        if self.aggr in [torch.max, torch.min]:
            x = self.aggr(x, dim=0)[0]
        elif self.aggr in [torch.mean, torch.sum]:
            x = self.aggr(x, dim=0)
        x = F.relu(F.linear(x, w3, b3.view(-1)))
        x = F.linear(x, w4, b4.view(-1))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
