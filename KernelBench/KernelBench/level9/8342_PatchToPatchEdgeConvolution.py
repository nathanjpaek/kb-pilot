import math
import torch
import torch.nn as nn
import torch.sparse as sp


class PatchToPatchEdgeConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super(PatchToPatchEdgeConvolution, self).__init__()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features,
            out_features))
        self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, edge_nodes, adj_matrix, inc_matrix, edge_feats):
        """
        :param edge_nodes: Matrix indicating the nodes which each edge in the
        batch connects. Shape [B, N]
        :param adj_matrix: Sparse adjacency matrix of the graph of shape
        [N, N]. Must contain only 1-entries (i.e. should not be normalised).
        :param inc_matrix: Sparse incidence matrix of the graph of shape
        [N, E].
        :param edge_feats: Features of *all* edges in the graph. Shape [E, D].
        :return: Hidden representation of shape [B, K].
        """
        batch_edge_idcs = sp.mm(adj_matrix.transpose(1, 0), edge_nodes.
            transpose(1, 0))
        batch_edge_idcs = sp.mm(inc_matrix.transpose(1, 0), batch_edge_idcs
            ).transpose(1, 0)
        batch_edge_idcs = (batch_edge_idcs == 2.0).float()
        row_sum = torch.sum(batch_edge_idcs, dim=1)
        inv = 1.0 / row_sum
        inv[torch.isinf(inv)] = 0.0
        batch_edge_idcs = batch_edge_idcs * inv.view(-1, 1)
        h_edges = torch.mm(edge_feats, self.weight) + self.bias
        h = torch.spmm(batch_edge_idcs, h_edges)
        return h


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
