import torch
import torch.nn as nn
from torch.nn import functional as F


class GATMutiHeadAttLayer(nn.Module):

    def __init__(self, in_features, out_features, heads, dropout=0.4, alpha
        =0.2, concat=True):
        super(GATMutiHeadAttLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.alpha = alpha
        self.concat = concat
        self.gain = nn.init.calculate_gain('leaky_relu', self.alpha)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.W = nn.Parameter(torch.zeros(size=(heads, in_features,
            out_features)))
        nn.init.xavier_uniform_(self.W.data, self.gain)
        self.a_1 = nn.Parameter(torch.zeros(size=(heads, out_features, 1)))
        nn.init.xavier_uniform_(self.a_1.data, self.gain)
        self.a_2 = nn.Parameter(torch.zeros(size=(heads, out_features, 1)))
        nn.init.xavier_uniform_(self.a_2.data, self.gain)

    def forward(self, input_seq, adj):
        if len(input_seq.size()) == 2:
            input_seq = torch.unsqueeze(input_seq, 0)
            adj = torch.unsqueeze(adj, 0)
        input_seq = torch.unsqueeze(input_seq, 1)
        adj = torch.unsqueeze(adj, 1)
        in_size = input_seq.size()
        nbatchs = in_size[0]
        slen = in_size[2]
        h = torch.matmul(input_seq, self.W)
        f_1 = torch.matmul(h, self.a_1)
        f_2 = torch.matmul(h, self.a_2)
        e = f_1.expand(nbatchs, self.heads, slen, slen) + f_2.expand(nbatchs,
            self.heads, slen, slen).transpose(2, 3)
        e = self.leakyrelu(e)
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj.expand(nbatchs, self.heads, slen, slen) >
            0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        node_out = torch.matmul(attention, h)
        if self.concat:
            node_out = node_out.transpose(1, 2).contiguous().view(nbatchs,
                slen, -1)
            node_out = F.elu(node_out)
        else:
            node_out = node_out.mean(1)
        return node_out.squeeze()


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MHAlayer1 = GATMutiHeadAttLayer(nfeat, nhid, nheads, dropout,
            alpha)
        self.out_att = GATMutiHeadAttLayer(nhid * nheads, nclass, 1,
            dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.MHAlayer1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5,
        'alpha': 4, 'nheads': 4}]
