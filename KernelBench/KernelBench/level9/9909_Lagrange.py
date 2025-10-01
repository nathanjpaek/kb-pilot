import torch
import torch.nn as nn
import torch.utils.data


def objective(x, h):
    return torch.log(1 + torch.sum(x * h, dim=1))


class Lagrange(nn.Module):

    def __init__(self):
        super(Lagrange, self).__init__()

    def forward(self, approx, dual, h):
        result = -objective(approx, h) + dual
        return torch.mean(result)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
