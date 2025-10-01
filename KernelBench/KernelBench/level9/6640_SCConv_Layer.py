import torch
import torch.nn as nn
import torch.nn.functional as F


class SCConv_Layer(nn.Module):

    def __init__(self, num_node_feats, num_edge_feats, num_triangle_feats,
        output_size, bias=True, f=F.relu):
        super().__init__()
        self.n2n_weights = nn.Linear(num_node_feats, output_size, bias=bias)
        self.n2e_weights = nn.Linear(num_node_feats, output_size, bias=bias)
        self.e2e_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
        self.e2n_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
        self.e2t_weights = nn.Linear(num_edge_feats, output_size, bias=bias)
        self.t2e_weights = nn.Linear(num_triangle_feats, output_size, bias=bias
            )
        self.t2t_weights = nn.Linear(num_triangle_feats, output_size, bias=bias
            )
        self.w = f

    def forward(self, X0, X1, X2, L0, L1, L2, B2D3, D2B1TD1inv, D1invB1,
        B2TD2inv):
        n2n = self.n2n_weights(X0)
        n2n = torch.sparse.mm(L0, n2n)
        n2e = self.n2e_weights(X0)
        n2e = torch.sparse.mm(D2B1TD1inv, n2e)
        e2n = self.e2n_weights(X1)
        e2n = torch.sparse.mm(D1invB1, e2n)
        e2e = self.e2e_weights(X1)
        e2e = torch.sparse.mm(L1, e2e)
        e2t = self.e2t_weights(X1)
        e2t = torch.sparse.mm(B2TD2inv, e2t)
        t2t = self.t2t_weights(X2)
        t2t = torch.sparse.mm(L2, t2t)
        t2e = self.t2e_weights(X2)
        t2e = torch.sparse.mm(B2D3, t2e)
        X0 = 1 / 2.0 * self.w(n2n + e2n)
        X1 = 1 / 3.0 * self.w(e2e + n2e + t2e)
        X2 = 1 / 2.0 * self.w(t2t + e2t)
        return X0, X1, X2


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.
        rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4,
        4])]


def get_init_inputs():
    return [[], {'num_node_feats': 4, 'num_edge_feats': 4,
        'num_triangle_feats': 4, 'output_size': 4}]
