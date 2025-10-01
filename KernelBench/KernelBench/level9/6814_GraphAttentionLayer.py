import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = Parameter(torch.FloatTensor(2 * output_dim, 1))
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.elu = nn.ELU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_normal_(self.a)

    def forward(self, input, adj):
        Wh = torch.mm(input, self.weight)
        Wh1 = torch.matmul(Wh, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_dim:, :])
        e = Wh1 + Wh2.T
        e = self.leakyrelu(e)
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        attention_wh = torch.matmul(attention, Wh)
        return self.elu(attention_wh)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'dropout': 0.5, 'alpha': 4}]
