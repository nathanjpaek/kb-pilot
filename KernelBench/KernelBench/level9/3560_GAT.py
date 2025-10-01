from torch.nn import Module
import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


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


class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0, alpha=0.2,
        concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        h.size()[0]
        a_input = (h.unsqueeze(0).repeat(h.shape[0], 1, 1) - h.unsqueeze(1)
            .repeat(1, h.shape[0], 1)).abs()
        e = F.relu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h = torch.matmul(attention, h)
        if self.concat:
            return F.relu(h)
        else:
            return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):

    def __init__(self, nfeat, nclass):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(nfeat, 128)
        self.gat2 = GATLayer(128, 128)
        self.gat3 = GATLayer(128, 128)
        self.edge_gc = EdgeGCN(128, nclass, include_adj=False)

    def forward(self, feat, adj):
        x = self.gat1(feat, adj)
        x = self.gat2(x, adj)
        x = self.gat3(x, adj)
        a = self.edge_gc(x, adj)
        return a


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nclass': 4}]
