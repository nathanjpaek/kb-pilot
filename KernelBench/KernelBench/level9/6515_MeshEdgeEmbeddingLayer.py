import torch
import torch.utils.data
import torch
from torch import nn


class MeshEdgeEmbeddingLayer(nn.Module):
    """
    Very important - who said that a-c is meaningfull at first layer...
    """

    def __init__(self, input_size, embedding_size, bias=True):
        super(MeshEdgeEmbeddingLayer, self).__init__()
        self.lin = nn.Linear(input_size, embedding_size, bias=bias)

    def forward(self, x):
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        return self.lin(x).permute(0, 2, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'embedding_size': 4}]
