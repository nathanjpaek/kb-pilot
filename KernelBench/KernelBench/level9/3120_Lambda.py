import torch
import torch.nn as nn


class Lambda(nn.Module):

    def forward(self, t, y):
        t = t.unsqueeze(0)
        equation = -1000 * y + 3000 - 2000 * torch.exp(-t)
        return equation


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
