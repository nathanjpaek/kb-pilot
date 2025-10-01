import torch
import torch.nn as nn


class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, nodes_feats, adj_matrix):
        num_neighbors = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(nodes_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbors
        return node_feats


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'c_in': 4, 'c_out': 4}]
