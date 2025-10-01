import torch
import torch.nn as nn


class SMAPELoss(nn.Module):

    def forward(self, input, target):
        return (torch.abs(input - target) / (torch.abs(input) + torch.abs(
            target) + 0.01)).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
