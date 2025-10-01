import torch
import torch.nn as nn


class SpatialGatingUnit(nn.Module):

    def __init__(self, dim_seq, dim_ff):
        super().__init__()
        self.proj = nn.Linear(dim_seq, dim_seq)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)
        self.norm = nn.LayerNorm(normalized_shape=dim_ff // 2, eps=1e-06)
        self.dim_ff = dim_ff
        self.activation = nn.GELU()

    def forward(self, x):
        res, gate = torch.split(tensor=x, split_size_or_sections=self.
            dim_ff // 2, dim=2)
        gate = self.norm(gate)
        gate = torch.transpose(gate, 1, 2)
        gate = self.proj(gate)
        gate = torch.transpose(gate, 1, 2)
        return gate * res


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_seq': 4, 'dim_ff': 4}]
