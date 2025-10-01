import torch
import torch.nn as nn


class GraphConvolution(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GraphConvolution, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, adj, nodes):
        nodes = torch.matmul(nodes, adj)
        nodes = self.relu(nodes)
        nodes = self.weight(nodes)
        nodes = self.relu(nodes)
        return nodes


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
