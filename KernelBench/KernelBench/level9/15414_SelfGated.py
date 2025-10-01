import torch
import torch.utils.data
import torch.nn.functional as F


class SelfGated(torch.nn.Module):
    """
    Self-Gated layer. math: \\sigmoid(W*x) * x
    """

    def __init__(self, input_size):
        super(SelfGated, self).__init__()
        self.linear_g = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        x_l = self.linear_g(x)
        x_gt = F.sigmoid(x_l)
        x = x * x_gt
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
