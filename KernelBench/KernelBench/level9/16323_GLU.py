import torch
import torch.nn as nn


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class GLU(nn.Module):

    def __init__(self, in_features, dropout_rate):
        super(GLU, self).__init__()
        self.sigm = nn.Sigmoid()
        self.W = nn.Linear(in_features, out_features=512, bias=True)
        self.V = nn.Linear(in_features, out_features=512, bias=True)
        initialize_weight(self.W)
        initialize_weight(self.V)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.W(x) * self.sigm(self.V(x))
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'dropout_rate': 0.5}]
