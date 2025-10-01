import torch
import torch.nn as nn


class My_loss_offset(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, mask, y, batch_size):
        return torch.sum(torch.abs(torch.pow(x - y, 2) * mask)
            ) / batch_size / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
