from torch.nn import Module
import torch
from torch.nn.modules.module import Module
import torch.nn as nn


class EdgeGCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, include_adj=True, bias=True):
        super(EdgeGCN, self).__init__()
        self.include_adj = include_adj
        self.in_features = in_features + 1 if self.include_adj else in_features
        self.out_features = out_features
        self.fc = nn.Linear(self.in_features, self.out_features, bias=bias)

    def forward(self, feat, adj):
        feat_diff = (feat.unsqueeze(0).repeat(feat.shape[0], 1, 1) - feat.
            unsqueeze(1).repeat(1, feat.shape[0], 1)).abs()
        if self.include_adj:
            x = torch.cat((feat_diff, adj.unsqueeze(2)), 2).view(feat.shape
                [0] * feat.shape[0], -1)
        else:
            x = feat_diff.view(feat.shape[0] * feat.shape[0], -1)
        output = self.fc(x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
