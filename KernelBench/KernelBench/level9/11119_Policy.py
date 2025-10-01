import torch
from copy import deepcopy
import torch.nn as nn


class Policy(nn.Module):

    def __init__(self, max_nodes, search_space):
        super(Policy, self).__init__()
        self.max_nodes = max_nodes
        self.search_space = deepcopy(search_space)
        self.edge2index = {}
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                self.edge2index[node_str] = len(self.edge2index)
        self.arch_parameters = nn.Parameter(0.001 * torch.randn(len(self.
            edge2index), len(search_space)))

    def generate_arch(self, actions):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = self.search_space[actions[self.edge2index[node_str]]]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.search_space[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def forward(self):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        return alphas


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'max_nodes': 4, 'search_space': [4, 4]}]
