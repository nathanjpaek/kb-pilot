import torch
import numpy as np
import torch.nn as nn


class ParallelLinear(nn.Module):

    def __init__(self, n_parallel, in_features, out_features, act=None,
        random_bias=False):
        super().__init__()
        self.act = act
        self.weight = nn.Parameter(torch.Tensor(n_parallel, in_features,
            out_features))
        self.bias = nn.Parameter(torch.Tensor(n_parallel, out_features))
        with torch.no_grad():
            self.weight.normal_(0.0, np.sqrt(2.0 / in_features))
            if random_bias:
                self.bias.normal_(0.0, np.sqrt(2.0 / in_features))
            else:
                self.bias.zero_()

    def forward(self, x):
        x = torch.bmm(x, self.weight) + self.bias[:, None, :]
        if self.act:
            x = self.act(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_parallel': 4, 'in_features': 4, 'out_features': 4}]
