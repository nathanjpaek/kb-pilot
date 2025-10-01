from torch.nn import Module
import torch
import torch.autograd
import torch.nn as nn
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907.
    """

    def __init__(self, state_dim, name='', out_state_dim=None):
        super(GraphConvolution, self).__init__()
        self.state_dim = state_dim
        if out_state_dim is None:
            self.out_state_dim = state_dim
        else:
            self.out_state_dim = out_state_dim
        self.fc1 = nn.Linear(in_features=self.state_dim, out_features=self.
            out_state_dim)
        self.fc2 = nn.Linear(in_features=self.state_dim, out_features=self.
            out_state_dim)
        self.name = name

    def forward(self, input, adj):
        state_in = self.fc1(input)
        forward_input = self.fc2(torch.bmm(adj, input))
        return state_in + forward_input


class GraphResConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907.
    """

    def __init__(self, state_dim, name=''):
        super(GraphResConvolution, self).__init__()
        self.state_dim = state_dim
        self.gcn_1 = GraphConvolution(state_dim, f'{name}_1')
        self.gcn_2 = GraphConvolution(state_dim, f'{name}_2')
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.name = name

    def forward(self, input, adj):
        output_1 = self.gcn_1(input, adj)
        output_1_relu = self.relu1(output_1)
        output_2 = self.gcn_2(output_1_relu, adj)
        output_2_res = output_2 + input
        output = self.relu2(output_2_res)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4}]
