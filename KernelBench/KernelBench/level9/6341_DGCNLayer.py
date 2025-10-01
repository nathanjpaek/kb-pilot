from _paritybench_helpers import _mock_config
from torch.nn import Module
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)
            )
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim
            ) * 2 * init_range - init_range
        return nn.Parameter(initial / 2)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.leakyrelu(self.gc1(x, adj))
        return x


class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """

    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt = opt
        self.dropout = opt['dropout']
        self.gc1 = GCN(nfeat=opt['feature_dim'], nhid=opt['hidden_dim'],
            dropout=opt['dropout'], alpha=opt['leakey'])
        self.gc2 = GCN(nfeat=opt['feature_dim'], nhid=opt['hidden_dim'],
            dropout=opt['dropout'], alpha=opt['leakey'])
        self.gc3 = GCN(nfeat=opt['hidden_dim'], nhid=opt['feature_dim'],
            dropout=opt['dropout'], alpha=opt['leakey'])
        self.gc4 = GCN(nfeat=opt['hidden_dim'], nhid=opt['feature_dim'],
            dropout=opt['dropout'], alpha=opt['leakey'])
        self.user_union = nn.Linear(opt['feature_dim'] + opt['feature_dim'],
            opt['feature_dim'])
        self.item_union = nn.Linear(opt['feature_dim'] + opt['feature_dim'],
            opt['feature_dim'])

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        Item_ho = self.gc2(vfea, UV_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return F.relu(User), F.relu(Item)

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)

    def forward_item(self, ufea, vfea, UV_adj, VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho = self.gc4(Item_ho, VU_adj)
        Item = torch.cat((Item_ho, vfea), dim=1)
        Item = self.item_union(Item)
        return F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho = self.gc3(User_ho, UV_adj)
        User = torch.cat((User_ho, ufea), dim=1)
        User = self.user_union(User)
        return F.relu(User)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config(dropout=0.5, feature_dim=4, hidden_dim
        =4, leakey=4)}]
