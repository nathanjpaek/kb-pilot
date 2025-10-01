import torch
from torch import nn


class ZeroOneTest(nn.Module):

    def __init__(self):
        super(ZeroOneTest, self).__init__()
        return

    def forward(self, output_p, output_n, prior):
        cost = prior * torch.mean((1 - torch.sign(output_p)) / 2)
        cost = cost + (1 - prior) * torch.mean((1 + torch.sign(output_n)) / 2)
        return cost


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
