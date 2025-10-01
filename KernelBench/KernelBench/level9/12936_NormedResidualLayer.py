from torch.nn import Module
import torch
from torch.nn import Dropout
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn.functional import gelu


class NormedResidualLayer(Module):

    def __init__(self, size, intermediate_size, dropout):
        super(NormedResidualLayer, self).__init__()
        self.mlp1 = Linear(size, intermediate_size, bias=True)
        self.mlp2 = Linear(intermediate_size, size, bias=True)
        self.layer_norm = LayerNorm((size,))
        self.dropout = Dropout(dropout)

    def forward(self, input):
        intermediate = gelu(self.mlp1(input))
        output = self.dropout(self.mlp2(intermediate)) + input
        output = self.layer_norm(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4, 'intermediate_size': 4, 'dropout': 0.5}]
