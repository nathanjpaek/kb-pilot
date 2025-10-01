import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, batch_num,
        node_num, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.batch_num = batch_num
        self.node_num = node_num
        torch.device('cuda')
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        :param h: (batch_size, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_size, number_nodes, out_features)
        """
        Wh = torch.matmul(h, self.W)
        e = self.prepare_batch(Wh)
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_size, number_nodes, out_features)
        :return:
        """
        _B, _N, _E = Wh.shape
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):

    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout, alpha,
        nheads, batch_num, node_num):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.batch_num = batch_num
        self.node_num = node_num
        torch.device('cuda')
        self.adj = nn.Parameter(torch.ones(self.batch_num, self.node_num,
            self.node_num))
        self.attentions = [GraphAttentionLayer(in_feat_dim, nhid, dropout=
            dropout, alpha=alpha, batch_num=batch_num, node_num=node_num,
            concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, out_feat_dim,
            dropout=dropout, alpha=alpha, batch_num=batch_num, node_num=
            node_num, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feat_dim': 4, 'nhid': 4, 'out_feat_dim': 4, 'dropout':
        0.5, 'alpha': 4, 'nheads': 4, 'batch_num': 4, 'node_num': 4}]
