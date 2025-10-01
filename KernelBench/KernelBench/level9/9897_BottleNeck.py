import torch
import torch.nn as nn


class BottleNeck(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        avg = x.mean(dim=-1).unsqueeze(2)
        return torch.cat((x, avg), dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
