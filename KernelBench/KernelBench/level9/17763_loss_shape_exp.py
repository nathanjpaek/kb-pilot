import torch
import torch.nn as nn


class loss_shape_exp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, beta=2):
        return torch.mean(torch.exp(beta * y) * torch.pow(x - y, 2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
