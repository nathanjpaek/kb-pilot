import torch
import torch.nn as nn


class Cat(nn.Module):

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
