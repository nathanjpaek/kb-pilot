import torch
import torch.nn as nn


class CustomizedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, y):
        return -torch.mean(torch.sum(output * y, dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
