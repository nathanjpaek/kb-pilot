import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha=0.2,
        concat=True, return_attention=False):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.return_attention = return_attention
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(
            in_features, out_features), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(
            out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(
            out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inputs, adj):
        h = torch.matmul(inputs, self.W)
        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(-2, -1))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return_vals = F.elu(h_prime)
        else:
            return_vals = h_prime
        if self.return_attention:
            return_vals = return_vals, attention
        return return_vals

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'dropout': 0.5}]
