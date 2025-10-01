import torch
import torch.nn as nn


def expand_toeplitz(diag, lower_diags, upper_diags):
    pattern = torch.cat([upper_diags, diag, lower_diags], 0)
    d = lower_diags.size(0)
    columns = []
    for i in range(d + 1):
        columns.append(pattern[d - i:d - i + d + 1])
    return torch.stack(columns, 0)


class ToeplitzBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.diag = nn.Parameter(torch.Tensor([0]))
        self.lower_diags = nn.Parameter(torch.Tensor(dim - 1).zero_())
        self.upper_diags = nn.Parameter(torch.Tensor(dim - 1).zero_())

    def diagonals(self):
        return [self.diag + 1, self.lower_diags, self.upper_diags]

    def forward(self, x):
        return torch.matmul(expand_toeplitz(*self.diagonals()), x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
