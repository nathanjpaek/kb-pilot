import torch
import torch.nn as nn
from torch.nn import Parameter


class EdgeGateFree(nn.Module):
    """
    Calculate gates for each edge in message passing.
    The gates are free parameters.
    Note:
        This will make the parameters depend on the number of edges, which will limit the model
        to work only on graphs with fixed number of edges.
    """

    def __init__(self, num_edges):
        super(EdgeGateFree, self).__init__()
        self.num_edges = num_edges
        self.edge_gates = Parameter(torch.Tensor(num_edges, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.edge_gates, 1)

    def forward(self, *args, **kwargs):
        return torch.sigmoid(self.edge_gates)


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'num_edges': 4}]
