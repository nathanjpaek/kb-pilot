import torch
from torch import nn


class AxialPositionalEmbedding(nn.Module):

    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i !=
            emb_dim_index]
        self.num_axials = len(shape)
        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape,
            ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'shape': [4, 4]}]
