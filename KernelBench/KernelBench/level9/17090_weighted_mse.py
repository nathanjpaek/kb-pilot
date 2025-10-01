import torch
from torch.nn.modules.loss import _Loss


class weighted_mse(_Loss):

    def __init__(self):
        super(weighted_mse, self).__init__()

    def forward(self, input, output, weight):
        return torch.sum(weight * (input - output) ** 2) / input.numel()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
