import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super(GraphConv, self).__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.w = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b
        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
