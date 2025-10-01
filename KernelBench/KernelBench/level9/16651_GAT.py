import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_feature, out_feature, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(self.in_feature, self.
            out_feature)))
        nn.init.xavier_normal(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.out_feature, 1)))
        nn.init.xavier_normal(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        h.size()[0]
        """
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        mask = -9e15 * (1.0 - adj)
        attention = e + mask


        #zero_vec = -9e15 * torch.ones_like(e)
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        """
        attn_for_self = torch.mm(h, self.a[0:self.out_feature, :])
        attn_for_neighs = torch.mm(h, self.a[self.out_feature:, :])
        dense = attn_for_self + attn_for_neighs.T
        dense = self.leakyrelu(dense)
        zero_vec = -9000000000000000.0 * torch.ones_like(dense)
        attention = torch.where(adj > 0, dense, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout,
            alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=
            dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5,
        'alpha': 4, 'nheads': 4}]
