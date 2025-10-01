import torch
import torch.nn as nn


class PartitionedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_c = nn.Linear(in_features // 2, out_features // 2, bias)
        self.linear_p = nn.Linear(in_features // 2, out_features // 2, bias)

    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)
        out_c = self.linear_c(x_c)
        out_p = self.linear_p(x_p)
        return out_c, out_p


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
