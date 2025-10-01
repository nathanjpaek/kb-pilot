import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn


class Discriminate_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(
            fake_outputs ** 2 / 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
