import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = self.dropout(input)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class InnerProductDecoder(nn.Module):
    """
    内积用来做decoder，用来生成邻接矩阵
    """

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        z = self.dropout(z)
        adj = torch.mm(z, z.t())
        return adj


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


class GATModelVAE(nn.Module):

    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,
        hidden_dim3, dropout, alpha, vae_bool=True):
        super(GATModelVAE, self).__init__()
        self.vae_bool = vae_bool
        self.gc_att = GraphAttentionLayer(input_feat_dim, hidden_dim1,
            dropout, alpha)
        self.gc1 = GraphConvolution(hidden_dim1, hidden_dim2, dropout)
        self.gc2 = GraphConvolution(hidden_dim2, hidden_dim3, dropout)
        self.gc3 = GraphConvolution(hidden_dim2, hidden_dim3, dropout)
        self.ip = InnerProductDecoder(dropout)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

    def encode(self, input, adj):
        gc_att = self.elu(self.gc_att(input, adj.to_dense()))
        hidden1 = self.relu(self.gc1(gc_att, adj))
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.vae_bool:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input, adj):
        mu, logvar = self.encode(input, adj)
        z = self.reparameterize(mu, logvar)
        return self.ip(z), mu, logvar


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_feat_dim': 4, 'hidden_dim1': 4, 'hidden_dim2': 4,
        'hidden_dim3': 4, 'dropout': 0.5, 'alpha': 4}]
