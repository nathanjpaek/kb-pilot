import torch
import torch.nn as nn


class Biaffine(nn.Module):

    def __init__(self, dim_left, dim_right):
        super().__init__()
        self.dim_left = dim_left
        self.dim_right = dim_right
        self.matrix = nn.Parameter(torch.Tensor(dim_left, dim_right))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.linear_l = nn.Linear(dim_left, 1)
        self.linear_r = nn.Linear(dim_right, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.matrix)

    def forward(self, x_l, x_r):
        x = torch.matmul(x_l, self.matrix)
        x = torch.bmm(x, x_r.transpose(1, 2)) + self.bias
        x += self.linear_l(x_l) + self.linear_r(x_r).transpose(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_left': 4, 'dim_right': 4}]
