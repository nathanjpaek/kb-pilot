import torch
import torch.nn.functional as F
import torch.autograd
import torch.nn as nn


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, input, adj):
        support = torch.mm(input, self.W)
        output = torch.mm(adj, support)
        return output


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x3 = self.gc2(x2, adj)
        return F.log_softmax(x3, dim=1), x2


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5}]
