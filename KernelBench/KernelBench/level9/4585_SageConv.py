from torch.nn import Module
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()
        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """
        if not isinstance(adj, torch.sparse.FloatTensor):
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (adj.sum(dim=1).
                    reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).
                    reshape(adj.shape[0], -1) + 1)
        else:
            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum
                (dim=1).reshape(adj.shape[0], -1) + 1)
        data = torch.cat([features, neigh_feature], dim=-1)
        combined = self.proj(data)
        return combined


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
