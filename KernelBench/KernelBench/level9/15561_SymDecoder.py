import torch
from torch import nn
import torch.utils.data


class SymDecoder(nn.Module):

    def __init__(self, feature_size, symmetry_size, hidden_size):
        super(SymDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.tanh = nn.Tanh()
        self.mlp_sg = nn.Linear(hidden_size, feature_size)
        self.mlp_sp = nn.Linear(hidden_size, symmetry_size)

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        sym_gen_vector = self.mlp_sg(vector)
        sym_gen_vector = self.tanh(sym_gen_vector)
        sym_param_vector = self.mlp_sp(vector)
        sym_param_vector = self.tanh(sym_param_vector)
        return sym_gen_vector, sym_param_vector


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'symmetry_size': 4, 'hidden_size': 4}]
