import math
import torch
import torch.nn as nn


def drop_none(**kwargs):
    r = {k: v for k, v in kwargs.items() if v is not None}
    return r


class L(nn.Module):

    def __init__(self, num_linear, input_features, output_features, dtype=
        None, device=None):
        super().__init__()
        options = drop_none(dtype=dtype, device=device)
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.empty((num_linear, input_features,
            output_features), **options))
        self.bias = nn.Parameter(torch.empty((num_linear, output_features),
            **options))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = torch.matmul(x.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            out += self.bias
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_linear': 4, 'input_features': 4, 'output_features': 4}]
