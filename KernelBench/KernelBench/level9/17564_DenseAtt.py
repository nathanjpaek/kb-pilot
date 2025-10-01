import torch
import torch.nn as nn
import torch.optim
import torch.nn.modules.loss


class DenseAtt(nn.Module):

    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward(self, x, adj):
        n = x.size(0)
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)
        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = torch.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'dropout': 0.5}]
