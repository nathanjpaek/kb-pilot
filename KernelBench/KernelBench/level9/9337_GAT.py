import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):

    def __init__(self, input_feature, output_feature, dropout, alpha,
        concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature))
            )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GAT(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout, alpha,
        nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout,
            alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=
            dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4,
        'dropout': 0.5, 'alpha': 4, 'nheads': 4}]
