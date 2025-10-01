import torch
import torch.nn as nn


class My_loss_focus2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y, batch_size):
        return torch.sum(torch.log1p(torch.abs(x - y))) / batch_size / 4


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
