import torch
import torch.nn as nn


class HGCN(nn.Module):

    def __init__(self, n_edges, in_feature, out_feature, n_agents):
        super(HGCN, self).__init__()
        None
        self.W_line = nn.Parameter(torch.ones(n_edges))
        self.W = None

    def forward(self, node_features, hyper_graph):
        self.W = torch.diag_embed(self.W_line)
        B_inv = torch.sum(hyper_graph.detach(), dim=-2)
        B_inv = torch.diag_embed(B_inv)
        softmax_w = torch.abs(self.W).detach()
        D_inv = torch.matmul(hyper_graph.detach(), softmax_w).sum(dim=-1)
        D_inv = torch.diag_embed(D_inv)
        D_inv = D_inv ** -0.5
        B_inv = B_inv ** -1
        D_inv[D_inv == float('inf')] = 0
        D_inv[D_inv == float('nan')] = 0
        B_inv[B_inv == float('inf')] = 0
        B_inv[B_inv == float('nan')] = 0
        A = torch.bmm(D_inv, hyper_graph)
        A = torch.matmul(A, torch.abs(self.W))
        A = torch.bmm(A, B_inv)
        A = torch.bmm(A, hyper_graph.transpose(-2, -1))
        A = torch.bmm(A, D_inv)
        X = torch.bmm(A, node_features)
        return X


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_edges': 4, 'in_feature': 4, 'out_feature': 4,
        'n_agents': 4}]
