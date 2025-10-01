from torch.nn import Module
import math
import torch
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Z = f(X, A) = softmax(A` * ReLU(A` * X * W0)* W1)
    A` = D'^(-0.5) * A * D'^(-0.5)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GCN(Module):
    """
    simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nfeat)
        self.gc2 = GraphConvolution(nfeat, nfeat)
        self.gc3 = GraphConvolution(nfeat, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        batch_size = adj.size(0)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)
        out = x.view(batch_size, -1)
        return out


class TripletGCN(Module):
    """
    Triplet network with GCN
    This GCN without any FC layer
    """

    def __init__(self, nfeat, dropout):
        super(TripletGCN, self).__init__()
        self.gcn = GCN(nfeat, dropout)

    def forward_once(self, x, adj):
        return self.gcn(x, adj)

    def forward(self, x1, adj1, x2, adj2, x3, adj3):
        out1 = self.forward_once(x1, adj1)
        out2 = self.forward_once(x2, adj2)
        out3 = self.forward_once(x3, adj3)
        dist_a = F.pairwise_distance(out1, out2, 2)
        dist_b = F.pairwise_distance(out1, out3, 2)
        return dist_a, dist_b


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'dropout': 0.5}]
