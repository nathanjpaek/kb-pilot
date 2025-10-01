import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn


class Adversarial_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs):
        return torch.mean((fake_outputs - 1) ** 2 / 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
