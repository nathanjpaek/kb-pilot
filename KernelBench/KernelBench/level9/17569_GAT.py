import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = Parameter(torch.empty(size=(in_features, out_features)))
        self.a = Parameter(torch.empty(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.shape[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks,
            Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features
            ), Wh_repeated_in_chunks, Wh_repeated_alternating

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        a_input, _Wh_repeated_in_chunks, _Wh_repeated_alternating = (self.
            _prepare_attentional_mechanism_input(Wh))
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return h_prime


class GAT(nn.Module):

    def __init__(self, node_feat, node_hid, dropout, alpha, nheads, concat=
        False):
        """Dense/multi-head version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.attentions = [GraphAttentionLayer(node_feat, node_hid, dropout
            =dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            y = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            y = torch.mean(torch.stack([att(x, adj) for att in self.
                attentions]), dim=0)
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.elu(y)
        return y


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'node_feat': 4, 'node_hid': 4, 'dropout': 0.5, 'alpha': 4,
        'nheads': 4}]
